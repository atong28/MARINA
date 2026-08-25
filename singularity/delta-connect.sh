#!/usr/bin/env bash
# Open a persistent, multiplexed SSH connection to NCSA Delta that an agent can reuse.
#
# You authenticate ONCE with your NCSA password + Duo. OpenSSH then keeps a control socket
# open, and every later command reuses that authenticated channel with no second Duo prompt --
# and, unlike driving a terminal with tmux send-keys, each command returns clean stdout/stderr
# and a real exit code.
#
# The master runs inside tmux so it survives your terminal closing, and so you can re-attach
# to re-authenticate when the session eventually expires. The socket path (~/.ssh/ncsa.sock)
# matches the `ncsa` host in ~/.ssh/config, so `ssh ncsa` ad hoc shares this same master.
#
# Delta's `login.delta…` name is round-robin across dt-login01/02 and tmux is node-local, so
# this pins dt-login01 -- reattaching to the wrong node would not find your session.
#
#   Usage:  bash singularity/delta-connect.sh <ncsa-username> [host]
#   Then:   ssh -S ~/.ssh/ncsa.sock <user>@<host> 'squeue -u <user>'
#   Status: bash singularity/delta-connect.sh --status <ncsa-username> [host]
#   Close:  bash singularity/delta-connect.sh --close  <ncsa-username> [host]
set -euo pipefail

SOCK="${NCSA_SOCK:-$HOME/.ssh/ncsa.sock}"
SESSION="${NCSA_TMUX_SESSION:-delta}"

MODE="connect"
case "${1:-}" in
  --status) MODE="status"; shift ;;
  --close)  MODE="close";  shift ;;
esac

USER_NAME="${1:-}"
HOST="${2:-dt-login01.delta.ncsa.illinois.edu}"
if [ -z "$USER_NAME" ]; then
  echo "usage: $0 [--status|--close] <ncsa-username> [host]" >&2
  exit 2
fi
TARGET="${USER_NAME}@${HOST}"
mkdir -p "$(dirname "$SOCK")"

case "$MODE" in
  status)
    if ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
      echo "socket ALIVE at $SOCK"
    else
      echo "socket DEAD (run without --status to open one)"; exit 1
    fi
    exit 0 ;;
  close)
    ssh -S "$SOCK" -O exit "$TARGET" 2>/dev/null || true
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "closed"; exit 0 ;;
esac

if ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
  echo "Already connected -- socket alive at $SOCK. Nothing to do."
  exit 0
fi

# Stale socket from a dead master would make ssh refuse to bind.
rm -f "$SOCK"
tmux kill-session -t "$SESSION" 2>/dev/null || true

# -M -N: act as master, run no remote command. ControlPersist keeps the socket usable
# briefly even if the master dies, so an in-flight command does not fail mid-run.
tmux new-session -d -s "$SESSION" \
  "ssh -M -S '$SOCK' -N \
      -o ControlPersist=600 \
      -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
      -o TCPKeepAlive=yes \
      '$TARGET'; \
   echo; echo '[master exited -- press enter to close]'; read _"

cat <<EOF

  tmux session '$SESSION' started with the SSH master to $TARGET.

  1. Attach and authenticate (NCSA password, then Duo -- type 1 for a push, or an
     app passcode):

       tmux attach -t $SESSION

  2. Once you are through Duo, detach with:  Ctrl-b  d
     (Do NOT Ctrl-c -- that kills the master and the socket with it.)

  3. Verify:

       bash $0 --status $USER_NAME $HOST

  After that, commands run over the existing authenticated channel, e.g.

       ssh -S $SOCK $TARGET 'squeue -u $USER_NAME'

EOF
