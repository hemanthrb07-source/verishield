#!/bin/bash
# VeriShield Development Startup Script
set -e

echo "🛡️  Starting VeriShield Development Environment"
echo "================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

echo ""
echo "📦 Installing Python dependencies..."
cd "$(dirname "$0")/.."
pip install -r backend/requirements.txt --quiet 2>/dev/null || pip install fastapi uvicorn python-multipart pydantic pydantic-settings sqlalchemy Pillow numpy torch torchvision scipy PyMuPDF python-dotenv httpx

echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent 2>/dev/null || echo "Using existing node_modules"
cd ..

echo ""
echo "🚀 Starting Backend (port 8000)..."
cd "$(dirname "$0")/.."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo ""
echo "⏳ Waiting for backend to start..."
sleep 3

echo ""
echo "🎨 Starting Frontend (port 5173)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ VeriShield is running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
