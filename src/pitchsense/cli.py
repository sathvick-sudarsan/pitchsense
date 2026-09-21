"""Small command-line entry point for package discovery."""

import argparse
import sys
from importlib.metadata import version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pitchsense")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('pitchsense')}"
    )
    arguments = sys.argv[1:] if argv is None else argv
    parser.parse_args(arguments)
    if not arguments:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
