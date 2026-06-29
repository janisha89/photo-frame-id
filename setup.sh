#!/bin/bash
# ─────────────────────────────────────────────────────────
#  Photo Frame Identification System — Setup Script (macOS)
# ─────────────────────────────────────────────────────────

echo ""
echo "  Photo Frame Identification System — Setup"
echo "─────────────────────────────────────────────"
echo ""

# 1. Check Python 3
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found."
  echo "    Download and install from: https://www.python.org/downloads/"
  exit 1
fi
echo "✓  Python 3: $(python3 --version)"

# 2. Upgrade pip silently
python3 -m pip install --upgrade pip -q

# 3. Try installing face_recognition directly
echo ""
echo "Installing packages (this can take 2–5 minutes the first time)…"
echo ""
pip3 install flask face-recognition numpy Pillow

# Check if it succeeded
if python3 -c "import face_recognition" &>/dev/null; then
  echo ""
  echo "✅  Setup complete!"
else
  echo ""
  echo "⚠️  face_recognition failed to install."
  echo "    This usually means 'dlib' needs cmake to build."
  echo ""
  echo "    Fix (macOS with Homebrew):"
  echo "      brew install cmake"
  echo "      pip3 install cmake dlib face-recognition"
  echo ""
  echo "    Or install Homebrew first from: https://brew.sh"
fi

echo ""
echo "─────────────────────────────────────────────"
echo "  To start the app:"
echo ""
echo "    python3 app.py"
echo ""
echo "  Then open in your browser:"
echo ""
echo "    http://localhost:5000"
echo "─────────────────────────────────────────────"
echo ""
