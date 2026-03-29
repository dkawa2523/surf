from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

CheckFn = Callable[[], tuple[bool, str]]
ImportCheckFn = Callable[[str], tuple[bool, str]]
EmitFn = Callable[[str], Any]


def run_required_module_checks(
    *,
    modules: Sequence[str],
    check_import: ImportCheckFn,
    emit: EmitFn = print,
) -> int:
    failures = 0
    for module_name in modules:
        ok, message = check_import(module_name)
        emit(message)
        if not ok:
            failures += 1
    return failures


def run_optional_module_checks(
    *,
    modules: Sequence[str],
    check_import: ImportCheckFn,
    full: bool,
    emit: EmitFn = print,
) -> int:
    failures = 0
    for module_name in modules:
        ok, message = check_import(module_name)
        if ok:
            emit(message)
            continue
        if full:
            emit(message)
            failures += 1
        else:
            emit(message.replace("fail:", "skip(optional):", 1))
    return failures


def run_named_check(
    *,
    fn: CheckFn,
    emit: EmitFn = print,
    count_skip_as_failure: bool = False,
) -> int:
    ok, message = fn()
    emit(message)
    if not ok:
        return 1
    if count_skip_as_failure and message.startswith("skip(optional):"):
        return 1
    return 0
