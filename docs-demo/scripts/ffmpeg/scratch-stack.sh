#!/usr/bin/env bash
# Boot an isolated stack for recording narrated demo videos — never the dev DB.
#
#   docs-demo/scripts/ffmpeg/scratch-stack.sh up          # backend :8031 + frontend :5176 on a fresh scratch DB
#   docs-demo/scripts/ffmpeg/scratch-stack.sh up --real   # + a scratch Celery worker; real train.py (videos 10–12)
#   docs-demo/scripts/ffmpeg/scratch-stack.sh down        # stop everything, delete the scratch dir
#
# Then record with DEMO_BASE_URL=http://localhost:5176 (record-narrated.sh sets it
# when the stack is up). Same env as CI's golden-path job: auth on (admin /
# sk-mock-admin-key bootstraps), simulated training, template synth + judge,
# HF offline. A fresh DB per `up`, so every recording starts from the same state.
# Simulated runs have no weights, so held-out eval / merge / quantize / chat fail
# on them; `--real` sets TRAINING_BACKEND=external and starts its own Celery
# worker on Redis DBs 4/5 (the dev worker uses 0/1 — never share a broker DB,
# the other worker would consume the task with the wrong DATA_DIR).
set -euo pipefail
REPO=/home/anuragj/Desktop/GitHub/__SLM__
SCRATCH=${DEMO_SCRATCH_DIR:-/tmp/brewslm-demo-stack}
BE_PORT=${DEMO_BACKEND_PORT:-8031}
FE_PORT=${DEMO_FRONTEND_PORT:-5176}
export PATH=$HOME/.nvm/versions/node/v22.23.2/bin:$PATH

wait_for() {
  for _ in $(seq 1 90); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done
  echo "timed out waiting for $1" >&2; return 1
}

case "${1:-}" in
  up)
    if ss -ltn | grep -qE ":($BE_PORT|$FE_PORT)\b"; then
      echo "port $BE_PORT or $FE_PORT already in use — run '$0 down' first" >&2; exit 1
    fi
    rm -rf "$SCRATCH"; mkdir -p "$SCRATCH/data"
    TRAINING=simulate
    [ "${2:-}" = "--real" ] && TRAINING=external
    export DATABASE_URL="sqlite+aiosqlite:///$SCRATCH/demo.db" DATA_DIR="$SCRATCH/data" \
      AUTH_ENABLED=true API_KEY=sk-mock-admin-key AUTH_BOOTSTRAP_API_KEY=sk-mock-admin-key \
      ALLOW_SIMULATED_TRAINING=true TRAINING_BACKEND=$TRAINING \
      ALLOW_SQLITE_AUTOCREATE=true DB_REQUIRE_ALEMBIC_HEAD=false \
      BREWSLM_TEMPLATE_SYNTH=1 EVAL_JUDGE_BACKEND=template HF_HUB_OFFLINE=1 \
      REDIS_URL=redis://localhost:6379/4 CELERY_BROKER_URL=redis://localhost:6379/4 \
      CELERY_RESULT_BACKEND=redis://localhost:6379/5
    ( cd "$REPO/backend"; \
      setsid nohup .venv/bin/uvicorn app.main:app --port "$BE_PORT" --no-access-log > "$SCRATCH/backend.log" 2>&1 < /dev/null &
      echo $! > "$SCRATCH/backend.pid" )
    if [ "$TRAINING" = external ]; then
      ( cd "$REPO/backend"; \
        setsid nohup .venv/bin/celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads \
          --concurrency=1 --hostname "demo-scratch@%h" > "$SCRATCH/worker.log" 2>&1 < /dev/null &
        echo $! > "$SCRATCH/worker.pid" )
    fi
    ( cd "$REPO/frontend"; \
      VITE_API_PROXY_TARGET="http://localhost:$BE_PORT" \
      setsid nohup npx vite --port "$FE_PORT" --strictPort > "$SCRATCH/frontend.log" 2>&1 < /dev/null &
      echo $! > "$SCRATCH/frontend.pid" )
    wait_for "http://localhost:$BE_PORT/api/health"
    wait_for "http://localhost:$FE_PORT/"
    curl -sf -o /dev/null "http://localhost:$FE_PORT/src/main.tsx" || true   # warm vite
    echo "scratch stack up ($TRAINING training): frontend http://localhost:$FE_PORT → backend :$BE_PORT, DB $SCRATCH/demo.db"
    ;;
  down)
    for f in backend frontend worker; do
      [ -f "$SCRATCH/$f.pid" ] && kill "$(cat "$SCRATCH/$f.pid")" 2>/dev/null || true
    done
    # uvicorn / vite children that outlived their parent on these ports
    for port in "$BE_PORT" "$FE_PORT"; do
      pid=$(ss -ltnp | grep -E ":$port\b" | grep -oP 'pid=\K[0-9]+' | head -1 || true)
      [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
      ss -ltn | grep -qE ":($BE_PORT|$FE_PORT)\b" || break
      sleep 0.5
    done
    rm -rf "$SCRATCH"
    echo "scratch stack down"
    ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
