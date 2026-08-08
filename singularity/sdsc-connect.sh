#!/usr/bin/env bash
# Open a persistent, multiplexed SSH connection to SDSC that an agent can reuse.
#
# You authenticate ONCE with password + TOTP. OpenSSH then keeps a control socket open,
# and every later command reuses that authenticated channel with no second prompt --
# and, unlike driving a terminal with tmux send-keys, each command returns clean
# stdout/stderr and a real exit code.
#
# The master runs inside tmux so it survives your terminal closing, and so you can
# re-attach to re-authenticate when the session eventually expires.
#
#   Usage:  bash singularity/sdsc-connect.sh <sdsc-username> [host]
#   Then:   ssh -S ~/.ssh/sdsc.sock <user>@<host> 'squeue -u <user>'
#   Status: bash singularity/sdsc-connect.sh --status <sdsc-username> [host]
#   Close:  bash singularity/sdsc-connect.sh --close  <sdsc-username> [host]
set -euo pipefail

SOCK="${SDSC_SOCK:-$HOME/.ssh/sdsc.sock}"
SESSION="${SDSC_TMUX_SESSION:-sdsc}"

MODE="connect"
case "${1:-}" in
  --status) MODE="status"; shift ;;
  --close)  MODE="close";  shift ;;
esac

USER_NAME="${1:-}"
HOST="${2:-login.expanse.sdsc.edu}"
if [ -z "$USER_NAME" ]; then
  echo "usage: $0 [--status|--close] <sdsc-username> [host]" >&2
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

  1. Attach and authenticate (password, then TOTP):

       tmux attach -t $SESSION

  2. Once you are through auth, detach with:  Ctrl-b  d
     (Do NOT Ctrl-c -- that kills the master and the socket with it.)

  3. Verify:

       bash $0 --status $USER_NAME $HOST

  After that, commands run over the existing authenticated channel, e.g.

       ssh -S $SOCK $TARGET 'squeue -u $USER_NAME'

EOF
