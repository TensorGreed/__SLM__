#!/usr/bin/env bash
# Isolated stack for tests/golden-path/main-path.spec.ts (real CPU training, so it needs a Celery worker).
#
#   bash tests/golden-path/main-path-stack.sh start          # prints STACK_DIR and E2E_BASE_URL
#   E2E_BASE_URL=http://localhost:5731 npm run test:e2e:main-path
#   bash tests/golden-path/main-path-stack.sh stop <STACK_DIR>
#
# Every start makes a BRAND-NEW stack dir (mktemp) holding a fresh scratch DB and a fresh, empty DATA_DIR. Never reuse
# a DATA_DIR: the fresh DB numbers its first project / run 1 again, and run 1 then picks up the old
# DATA_DIR/projects/1/experiments/1 directory (an old adapter in it marks the new run completed at once, and the eval
# and export read the old weights while new training is still writing).
#
# Own ports (BACKEND_PORT 8731, VITE_PORT 5731) and own Redis dbs (REDIS_DBS "13 14": broker + results), refused if busy
# or non-empty, flushed on stop. CUDA_VISIBLE_DEVICES="" keeps training on CPU. Never touches the dev stack (uvicorn
# :8000, its celery, vite :5173) or data/slm_platform.db. Needs SmolLM2-135M-Instruct in the HF cache.
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY="${BACKEND_PY:-$REPO/backend/.venv/bin/python}"
BACKEND_PORT="${BACKEND_PORT:-8731}"
VITE_PORT="${VITE_PORT:-5731}"
read -r BROKER_DB RESULT_DB <<<"${REDIS_DBS:-13 14}"
REDIS_HOST="${REDIS_HOST:-localhost}"

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

redis_keys() {
    "$PY" -c "import redis,sys; print(redis.Redis(host='$REDIS_HOST', db=int(sys.argv[1])).dbsize())" "$1"
}

start() {
    for p in "$BACKEND_PORT" "$VITE_PORT"; do
        if port_busy "$p"; then echo "port $p is in use; pick another (BACKEND_PORT / VITE_PORT)" >&2; exit 1; fi
    done
    for db in "$BROKER_DB" "$RESULT_DB"; do
        if [ "$(redis_keys "$db")" != "0" ]; then echo "redis db $db is not empty; pick unused dbs (REDIS_DBS)" >&2; exit 1; fi
    done
    STACK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/brewslm-main-path-XXXXXX")"
    DATA_DIR="$STACK_DIR/data"   # brand new and empty, every run
    mkdir "$DATA_DIR"
    export DATA_DIR DATABASE_URL="sqlite+aiosqlite:///$STACK_DIR/scratch.db"
    export AUTH_ENABLED=true API_KEY=sk-mock-admin-key AUTH_BOOTSTRAP_API_KEY=sk-mock-admin-key
    export ALLOW_SQLITE_AUTOCREATE=true DB_REQUIRE_ALEMBIC_HEAD=false TRAINING_BACKEND=external
    export HF_HUB_OFFLINE=1 AUTO_LIFT_EVAL_ENABLED=false CUDA_VISIBLE_DEVICES=""
    export REDIS_URL="redis://$REDIS_HOST:6379/$BROKER_DB"
    export CELERY_BROKER_URL="redis://$REDIS_HOST:6379/$BROKER_DB" CELERY_RESULT_BACKEND="redis://$REDIS_HOST:6379/$RESULT_DB"

    # Each process leads its own session (setsid), so stop can signal its whole group by pid.
    (cd "$REPO/backend"; setsid "$PY" -m uvicorn app.main:app --port "$BACKEND_PORT" --no-access-log \
        >"$STACK_DIR/backend.log" 2>&1 & echo $! >"$STACK_DIR/backend.pid")
    (cd "$REPO/backend"; setsid "$PY" -m celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads \
        --concurrency=2 >"$STACK_DIR/worker.log" 2>&1 & echo $! >"$STACK_DIR/worker.pid")
    (cd "$REPO/frontend"; VITE_API_PROXY_TARGET="http://localhost:$BACKEND_PORT" setsid npx vite --port "$VITE_PORT" \
        --strictPort >"$STACK_DIR/vite.log" 2>&1 & echo $! >"$STACK_DIR/vite.pid")
    echo "$BACKEND_PORT $VITE_PORT $BROKER_DB $RESULT_DB" >"$STACK_DIR/stack.conf"

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null && port_busy "$VITE_PORT"; then
            echo "STACK_DIR=$STACK_DIR"
            echo "E2E_BASE_URL=http://localhost:$VITE_PORT"
            return 0
        fi
        sleep 1
    done
    echo "stack did not come up; logs in $STACK_DIR" >&2
    exit 1
}

stop() {
    STACK_DIR="${1:?usage: stop <STACK_DIR>}"
    read -r BACKEND_PORT VITE_PORT BROKER_DB RESULT_DB <"$STACK_DIR/stack.conf"
    for name in vite worker backend; do
        pid="$(cat "$STACK_DIR/$name.pid" 2>/dev/null || true)"
        [ -n "$pid" ] || continue
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for name in vite worker backend; do
        pid="$(cat "$STACK_DIR/$name.pid" 2>/dev/null || true)"
        [ -n "$pid" ] && kill -KILL -- "-$pid" 2>/dev/null || true
    done
    for db in "$BROKER_DB" "$RESULT_DB"; do
        "$PY" -c "import redis,sys; redis.Redis(host='$REDIS_HOST', db=int(sys.argv[1])).flushdb()" "$db"
    done
    for p in "$BACKEND_PORT" "$VITE_PORT"; do
        if port_busy "$p"; then echo "port $p still in use" >&2; else echo "port $p free"; fi
    done
    echo "redis dbs $BROKER_DB $RESULT_DB flushed ($(redis_keys "$BROKER_DB") / $(redis_keys "$RESULT_DB") keys); logs kept in $STACK_DIR"
}

case "${1:-}" in
    start) start ;;
    stop) shift; stop "$@" ;;
    *) echo "usage: $0 start | stop <STACK_DIR>" >&2; exit 2 ;;
esac
