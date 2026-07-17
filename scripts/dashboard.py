from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import run_streamlit_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Streamlit dashboard.")
    parser.add_argument(
        "--app",
        default=str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"),
        help="Dashboard app path.",
    )
    parser.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_streamlit_app(Path(args.app), args.streamlit_args))


if __name__ == "__main__":
    main()