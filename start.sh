#!/bin/bash
# ava_backtest stack startup
# Brings up TimescaleDB + Redis + FastAPI, then starts the UI dev server
# Usage: ./start.sh [--down] [--logs]
set -e

API_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="/home/noufal/projects/ava-backtest-ui"
LOG_DIR="$API_DIR/logs"
UI_LOG="$LOG_DIR/ui.log"
UI_PID="$LOG_DIR/ui.pid"

mkdir -p "$LOG_DIR"

# ── Flags ─────────────────────────────────────────────────────────────────────
if [[ "$1" == "--down" ]]; then
  echo "▼  Stopping stack..."
  cd "$API_DIR" && docker-compose down
  [[ -f "$UI_PID" ]] && kill "$(cat $UI_PID)" 2>/dev/null && rm "$UI_PID" && echo "UI stopped"
  exit 0
fi

if [[ "$1" == "--logs" ]]; then
  echo "=== API (docker) ===" && docker-compose -f "$API_DIR/docker-compose.yml" logs --tail=40
  echo "=== UI ===" && tail -40 "$UI_LOG" 2>/dev/null
  exit 0
fi

# ── Backend ───────────────────────────────────────────────────────────────────
echo "▲  Starting backend (TimescaleDB + Redis + FastAPI)..."
cd "$API_DIR"
docker-compose up -d --build

echo -n "   Waiting for API to be healthy"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8201/health > /dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 2
done

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "▲  Starting frontend (Vite dev server on :8203)..."
cd "$UI_DIR"

# Kill existing UI process if any
[[ -f "$UI_PID" ]] && kill "$(cat $UI_PID)" 2>/dev/null || true

npm run dev -- --host > "$UI_LOG" 2>&1 &
echo $! > "$UI_PID"
sleep 3

if kill -0 "$(cat $UI_PID)" 2>/dev/null; then
  echo "   UI running (pid $(cat $UI_PID))"
else
  echo "   ❌ UI failed to start — check $UI_LOG"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
TAILSCALE_IP="100.90.137.65"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AvaAI Backtest — RUNNING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UI  (Tailscale) → http://$TAILSCALE_IP:8203"
echo "  API (Tailscale) → http://$TAILSCALE_IP:8201"
echo "  API docs        → http://$TAILSCALE_IP:8201/docs"
echo "  DB              → postgresql://ava@$TAILSCALE_IP:5435/ava"
echo "  Logs            → ./start.sh --logs"
echo "  Stop            → ./start.sh --down"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
