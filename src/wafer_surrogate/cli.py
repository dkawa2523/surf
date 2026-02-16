from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from wafer_surrogate.cli_parser import build_parser as _build_parser


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    return args[1:] if args[:1] == ["--"] else args


def build_parser() -> argparse.ArgumentParser:
    return _build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    handler = getattr(args, "handler", None)
    if handler is None:
        handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
