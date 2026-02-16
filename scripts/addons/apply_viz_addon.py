#!/usr/bin/env python3
"""
apply_viz_addon.py

- Merge visualization add-on tasks into tasks/tasks.json.
- Optionally append VIZ requirement/traceability notes (non-destructive).
- Regenerate tasks/TASKS.md list for convenience.

This script is designed to be safe:
- It creates timestamped backups for each modified file.
- It refuses to overwrite if task_id conflicts are detected.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _backup(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + f".bak.{_now_tag()}")
    shutil.copy2(path, bak)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_section_if_missing(path: Path, marker: str, content: str) -> bool:
    if not path.exists():
        return False
    txt = path.read_text(encoding="utf-8")
    if marker in txt:
        return False
    _backup(path)
    path.write_text(txt.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")
    return True


def _regenerate_tasks_md(tasks_json: Dict[str, Any], out_path: Path) -> None:
    # Group by phase in order of appearance
    phase_order: List[str] = []
    phase_to_items: Dict[str, List[str]] = {}
    for t in tasks_json.get("tasks", []):
        phase = str(t.get("phase", ""))
        if phase not in phase_to_items:
            phase_to_items[phase] = []
            phase_order.append(phase)
        phase_to_items[phase].append(f"- {t['task_id']}: {t.get('title','')}".rstrip())

    lines: List[str] = []
    lines.append("# TASKS")
    lines.append("")
    lines.append("`tasks/tasks.json` がSSOTです。この文書は一覧用（生成）。")
    lines.append("")

    for ph in phase_order:
        if not ph:
            header = "## (no phase)"
        else:
            header = f"## {ph}"
        lines.append(header)
        lines.append("")
        lines.extend(phase_to_items.get(ph, []))
        lines.append("")

    _backup(out_path)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # scripts/addons -> scripts -> repo
    tasks_path = repo_root / "tasks" / "tasks.json"
    fragment_path = Path(__file__).with_name("viz_tasks_fragment.json")

    if not tasks_path.exists():
        print(f"[ERROR] tasks file not found: {tasks_path}", file=sys.stderr)
        return 2
    if not fragment_path.exists():
        print(f"[ERROR] fragment file not found: {fragment_path}", file=sys.stderr)
        return 2

    tasks_json = _load_json(tasks_path)
    frag = _load_json(fragment_path)

    insert_after = frag.get("insert_after_task_id", "P0-CHECKPOINT")
    insert_before = frag.get("insert_before_task_id", "DECISION-P1-001")
    new_tasks: List[Dict[str, Any]] = frag.get("tasks", [])

    existing_ids = {t["task_id"] for t in tasks_json.get("tasks", [])}
    for t in new_tasks:
        if t["task_id"] in existing_ids:
            print(f"[ERROR] task_id conflict: {t['task_id']}", file=sys.stderr)
            print("Refusing to merge. Please rename task_id or remove duplicate first.", file=sys.stderr)
            return 3

    # Find insertion point
    tasks_list: List[Dict[str, Any]] = list(tasks_json.get("tasks", []))
    idx_after = next((i for i, t in enumerate(tasks_list) if t.get("task_id") == insert_after), None)
    idx_before = next((i for i, t in enumerate(tasks_list) if t.get("task_id") == insert_before), None)

    if idx_after is None:
        # append at end
        insert_pos = len(tasks_list)
    else:
        insert_pos = idx_after + 1

    if idx_before is not None:
        # ensure we insert before the before-task (even if after-task not found)
        insert_pos = min(insert_pos, idx_before)

    _backup(tasks_path)
    tasks_list[insert_pos:insert_pos] = new_tasks
    tasks_json["tasks"] = tasks_list

    # ensure policy_lock_text exists for compatibility
    if "policy_lock_text" not in tasks_json and "policy_lock" in tasks_json:
        tasks_json["policy_lock_text"] = tasks_json["policy_lock"]

    _write_json(tasks_path, tasks_json)
    print(f"[OK] merged {len(new_tasks)} tasks into {tasks_path}")

    # Regenerate tasks/TASKS.md (best-effort)
    tasks_md_path = repo_root / "tasks" / "TASKS.md"
    if tasks_md_path.exists():
        _regenerate_tasks_md(tasks_json, tasks_md_path)
        print(f"[OK] regenerated {tasks_md_path}")

    # Optional: append requirements/traceability notes (non-destructive)
    req_path = repo_root / "docs" / "REQUIREMENTS.md"
    tr_path = repo_root / "docs" / "TRACEABILITY.md"
    # We append the separate addon markdown with a marker, to avoid risky edits.
    addon_req = repo_root / "docs" / "VIZ_REQUIREMENTS_ADDON.md"
    addon_tr = repo_root / "docs" / "VIZ_TRACEABILITY_ADDON.md"
    # If addon docs exist (copied by unzip), offer to append.
    if addon_req.exists() and req_path.exists():
        marker = "## VIZ add-on"
        content = "## VIZ add-on\n\n（注）詳細は `docs/VIZ_REQUIREMENTS_ADDON.md` を参照。"
        appended = _append_section_if_missing(req_path, marker, content)
        if appended:
            print(f"[OK] appended marker section into {req_path}")

    if addon_tr.exists() and tr_path.exists():
        marker = "R-SHOULD-VIZ-001"
        content = addon_tr.read_text(encoding="utf-8").strip()
        appended = _append_section_if_missing(tr_path, marker, content)
        if appended:
            print(f"[OK] appended VIZ traceability into {tr_path}")

    print("[DONE] apply_viz_addon completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
