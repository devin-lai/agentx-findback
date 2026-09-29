#!/usr/bin/env bash
# User-level process control for the assigned node. No sudo or system configuration.
set -euo pipefail
cd "$(dirname "$0")/../.."
session="agentx-findback"
command -v tmux >/dev/null || { echo "tmux is required in the approved environment."; exit 1; }
case "${1:-status}" in
  start)
    test -x .venv/bin/agentx || { echo "Prepare .venv in the approved runtime first."; exit 1; }
    .venv/bin/python scripts/ops/preflight.py --require-cuda
    if tmux has-session -t "=$session" 2>/dev/null; then
      echo "AgentX session already exists; inspect it before restarting."
      exit 1
    fi
    # The node application binds only loopback; reach it through an SSH tunnel.
    # Cached weights only: a node deployment never downloads models at runtime.
    tmux new-session -d -s "$session" -c "$PWD" \
      "HF_HUB_OFFLINE=${AGENTX_HF_OFFLINE:-1} PATH=$PWD/.venv/bin:\$PATH .venv/bin/agentx serve --host 127.0.0.1 --port 9000"
    echo "Started user tmux session: $session. Check /api/health through your SSH tunnel."
    ;;
  status)
    tmux has-session -t "=$session" 2>/dev/null && echo "AgentX tmux session exists." || { echo "AgentX session is not running."; exit 1; }
    ;;
  stop)
    # Graceful interrupt only our named session; never kill unrelated Python processes.
    tmux has-session -t "=$session" 2>/dev/null || { echo "AgentX session is not running."; exit 0; }
    tmux send-keys -t "=$session:" C-c
    for i in $(seq 1 45); do
      tmux has-session -t "=$session" 2>/dev/null || { echo "AgentX stopped."; exit 0; }
      # A second interrupt asks Uvicorn to force-exit if a shutdown is stuck on open connections.
      [ "$i" -eq 20 ] && tmux send-keys -t "=$session:" C-c
      # Last resort for this project's own session only: hang up the pane (SIGHUP to the server).
      [ "$i" -eq 35 ] && tmux kill-session -t "=$session:"
      sleep 1
    done
    echo "AgentX session could not be stopped; inspect it manually."
    exit 1
    ;;
  *) echo "Usage: scripts/ops/node_service.sh start|status|stop"; exit 2 ;;
esac
