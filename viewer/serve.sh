#!/bin/bash
PORT="${1:-8080}"
URL="http://localhost:$PORT"
cd "$(dirname "$0")"
mkdir -p splats

echo "H2102 Viewer: $URL"

# Auto-launch browser after server starts (background, give server a moment)
if [ -n "$BROWSER" ]; then
    (sleep 1 && "$BROWSER" "$URL") &
elif command -v xdg-open &>/dev/null; then
    (sleep 1 && xdg-open "$URL") &
fi

python3 -m http.server "$PORT"
