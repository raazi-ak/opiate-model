#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/raazifaisal/Documents/Opiate-model"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend-nextjs"
VENV_ACTIVATE="$ROOT/.venv/bin/activate"
RUN_DIR="$ROOT/.run"
LOG_DIR="$ROOT/.logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

backend_pid_file="$RUN_DIR/backend.pid"
frontend_pid_file="$RUN_DIR/frontend.pid"

start_backend() {
  echo "Starting backend on :8000 ..."
  # Best-effort kill anything already on 8000
  lsof -ti:8000 | xargs kill -9 >/dev/null 2>&1 || true
  nohup bash -lc "source '$VENV_ACTIVATE'; cd '$BACKEND_DIR'; python -m app.main" \
    >"$LOG_DIR/backend.out" 2>"$LOG_DIR/backend.err" &
  echo $! > "$backend_pid_file"
  echo "Backend PID $(cat "$backend_pid_file") (logs: $LOG_DIR/backend.*)"
}

start_frontend() {
  echo "Starting frontend on :3000 ..."
  # Best-effort kill anything already on 3000
  lsof -ti:3000 | xargs kill -9 >/dev/null 2>&1 || true
  nohup bash -lc "cd '$FRONTEND_DIR'; npm run dev" \
    >"$LOG_DIR/frontend.out" 2>"$LOG_DIR/frontend.err" &
  echo $! > "$frontend_pid_file"
  echo "Frontend PID $(cat "$frontend_pid_file") (logs: $LOG_DIR/frontend.*)"
}

stop_backend() {
  if [[ -f "$backend_pid_file" ]]; then
    kill -9 "$(cat "$backend_pid_file")" >/dev/null 2>&1 || true
    rm -f "$backend_pid_file"
  fi
  lsof -ti:8000 | xargs kill -9 >/dev/null 2>&1 || true
  echo "Backend stopped."
}

stop_frontend() {
  if [[ -f "$frontend_pid_file" ]]; then
    kill -9 "$(cat "$frontend_pid_file")" >/dev/null 2>&1 || true
    rm -f "$frontend_pid_file"
  fi
  lsof -ti:3000 | xargs kill -9 >/dev/null 2>&1 || true
  echo "Frontend stopped."
}

status_all() {
  echo "Backend:  $(lsof -ti:8000 | wc -l | tr -d ' ') listening on 8000"
  echo "Frontend: $(lsof -ti:3000 | wc -l | tr -d ' ') listening on 3000"
  [[ -f "$backend_pid_file" ]] && echo "Backend PID file: $(cat "$backend_pid_file")" || true
  [[ -f "$frontend_pid_file" ]] && echo "Frontend PID file: $(cat "$frontend_pid_file")" || true
}

usage() {
  cat <<EOF
Usage: ./dev.sh <command>

Commands:
  start           Start backend and frontend
  stop            Stop backend and frontend
  restart         Restart both
  status          Show port/PID status
  start-backend   Start backend only
  stop-backend    Stop backend only
  start-frontend  Start frontend only
  stop-frontend   Stop frontend only

Logs:
  $LOG_DIR/backend.out|err
  $LOG_DIR/frontend.out|err
EOF
}

cmd="${1:-}" || true
case "$cmd" in
  start)
    start_backend
    start_frontend
    ;;
  stop)
    stop_backend
    stop_frontend
    ;;
  restart)
    stop_backend
    stop_frontend
    start_backend
    start_frontend
    ;;
  status)
    status_all
    ;;
  start-backend)
    start_backend
    ;;
  stop-backend)
    stop_backend
    ;;
  start-frontend)
    start_frontend
    ;;
  stop-frontend)
    stop_frontend
    ;;
  *)
    usage
    ;;
esac


