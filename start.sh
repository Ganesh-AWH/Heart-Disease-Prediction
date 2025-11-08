#!/bin/bash

echo "================================"
echo "Heart Disease Prediction System"
echo "================================"
echo ""
echo "Starting backend server on port 5000..."
echo "Starting frontend dev server on port 5173..."
echo ""
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:5003"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "================================"
echo ""

python app.py &
BACKEND_PID=$!

npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

wait
