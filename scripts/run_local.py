"""
Quick local runner — starts the API without Docker.
Usage: python scripts/run_local.py
       python scripts/run_local.py --port 8080 --reload
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run Crop Disease Classifier locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="./models/best_model.pth")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    if not Path(args.model).exists():
        print(f"[WARNING] Model not found at {args.model}. Starting in demo mode.")
        print("   To train: python src/train.py --data ./data/plantvillage")
    else:
        print(f"[OK] Model: {args.model}")

    print(f"\nStarting Crop Disease Classifier at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop.\n")

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
