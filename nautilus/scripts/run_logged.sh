#!/usr/bin/env bash
# Run a command with its output persisted off the pod.
#
#   bash /root/gurusmart/run_logged.sh <command> [args...]
#
# Pod logs vanish when the pod is garbage-collected and events expire within the hour, so a
# job that dies overnight leaves nothing to diagnose. This tees everything to the PVC and,
# critically, records SIGTERM -- Kubernetes sends it before killing an evicted or preempted
# pod, so its presence in the log is the difference between "the run crashed" and "the
# cluster took the node away". Without it you are guessing from timestamps.
#
# The log path is stable across retries and appends, so with backoffLimit > 0 every attempt
# lands in one file in order.
#
# Env:
#   SMART_RUN_ID   names the log directory (same var that pins the results dir). Required
#                  in practice; falls back to the pod hostname.
#   SMART_LOG_DIR  overrides the directory outright.
set -uo pipefail

RUN_NAME="${SMART_RUN_ID:-${HOSTNAME:-unnamed}}"
LOGDIR="${SMART_LOG_DIR:-/root/gurusmart/results/${RUN_NAME}}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pod.log"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

{
  echo
  echo "======================================================================"
  echo "[wrapper] attempt start   $(stamp)"
  echo "[wrapper] pod=${HOSTNAME:-?}  node=${NODE_NAME:-?}"
  echo "[wrapper] cmd: $*"
  echo "======================================================================"
} >> "$LOG"

# Everything below is duplicated to the log. Done with exec so the redirect also covers
# anything the command spawns, not just its own stdout.
exec > >(tee -a "$LOG") 2>&1

got_term=0
on_term() {
  got_term=1
  echo "[wrapper] !!! SIGTERM at $(stamp) -- pod is being evicted/preempted, not a crash"
}
trap on_term TERM
trap 'echo "[wrapper] !!! SIGINT at $(stamp)"' INT

# Run in the background and wait, rather than in the foreground: bash defers trap handlers
# until the current foreground command finishes, so a foreground child would swallow the
# SIGTERM notice until after the kill -- which is exactly when it stops being useful.
"$@" &
child=$!
wait "$child"
rc=$?

if [ "$got_term" -eq 1 ]; then
  echo "[wrapper] attempt end     $(stamp)  rc=$rc  (terminated by cluster)"
else
  echo "[wrapper] attempt end     $(stamp)  rc=$rc"
fi

# Give tee a moment to flush before the pod goes away.
sleep 1
exit "$rc"
