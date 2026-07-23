from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Streamlit dashboard."
    )

    parser.add_argument(
        "--app",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "src"
            / "dashboard"
            / "app.py"
        ),
        help="Path to the Streamlit application.",
    )

    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def run_streamlit(app: Path, extra_args: list[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        *extra_args,
    ]

    return subprocess.call(command)


def main() -> None:
    args = parse_args()

    raise SystemExit(
        run_streamlit(
            app=args.app,
            extra_args=args.streamlit_args,
        )
    )


if __name__ == "__main__":
    main()