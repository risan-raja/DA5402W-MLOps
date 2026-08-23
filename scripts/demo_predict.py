"""CLI wrapper: python scripts/demo_predict.py  (or: python -m src.deployment.demo)."""

from src.deployment.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
