"""PitchSense command-line entry point."""

import argparse
import logging
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from pitchsense.config import PipelineConfig, load_pipeline_config
from pitchsense.detection.base import Detector
from pitchsense.errors import BackendUnavailableError, PitchSenseError
from pitchsense.logging import configure_logging
from pitchsense.pipeline import PipelineRunner
from pitchsense.tracking.base import Tracker


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[PipelineConfig], tuple[Detector, Tracker]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="pitchsense")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('pitchsense')}"
    )
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="process a video")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--model")
    run.add_argument("--device")
    run.add_argument("--confidence", type=float)
    run.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if not arguments:
        parser.print_help()
    if args.command == "run":
        try:
            config = load_pipeline_config(
                args.config,
                model=args.model,
                device=args.device,
                confidence=args.confidence,
            )
            configure_logging(getattr(logging, args.log_level))
            if backend_factory is None:
                raise BackendUnavailableError(
                    "production detector and tracker are unavailable until PR3"
                )
            detector, tracker = backend_factory(config)
            PipelineRunner(detector, tracker, config).run(args.input, args.output)
        except PitchSenseError as exc:
            print(f"pitchsense: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
