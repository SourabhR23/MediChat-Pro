#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait until backend is healthy before starting frontend
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is up!"
        break
    fi
    echo "Attempt $i/30 — waiting..."
    sleep 2
done

echo "Starting Streamlit frontend on port $PORT..."
API_BASE_URL=http://localhost:8000 streamlit run main2.py \
    --server.address 0.0.0.0 \
    --server.port $PORT \
    --server.headless true

# If streamlit exits, kill backend too
kill $BACKEND_PID
