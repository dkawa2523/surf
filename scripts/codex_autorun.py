#!/usr/bin/env python3
"""Codex autorun runner.

- Reads tasks/tasks.json
- Executes each task via `codex exec` with robust flag detection
- Records completed tasks to runs/autorun_state.json
- Stops on checkpoint/decision with exit code 42

Design goals:
- Never restart from scratch; always resume from state
- Avoid read-only execution by auto-detecting write flags
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT_DIR / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TASKS_PATH = ROOT_DIR / "tasks" / "tasks.json"
PROMPTS_DIR = ROOT_DIR / "prompts"
STATE_PATH = RUNS_DIR / "autorun_state.json"
LOG_DIR = RUNS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STOP_CODE = 42


@dataclass
class Task:
    task_id: str
    phase: str
    type: str
    title: str
    description: str
    acceptance_criteria: List[str]
    estimated_minutes: int
    scope_limits: Dict[str, Any]
    verification_commands: List[str]
    stop_after: bool


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt state should not crash; start fresh but keep file for forensics
            return {"completed": [], "history": []}
    return {"completed": [], "history": []}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


def which(cmd: str) -> Optional[str]:
    from shutil import which as _which

    return _which(cmd)


def run_capture(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err


def detect_codex_exec_capabilities() -> Dict[str, Any]:
    """Detect flags supported by `codex exec`.

    We only rely on parsing help output; we do NOT assume any specific CLI version.
    """
    caps: Dict[str, Any] = {
        "has_model": False,
        "has_reasoning": False,
        "has_skip_git_repo_check": False,
        "has_sandbox": False,
        "write_flag": None,
        "has_prompt": False,
        "has_prompt_file": False,
        "json_mode": None,
    }

    code, out, err = run_capture(["codex", "exec", "--help"])
    help_text = out + "\n" + err
    if code != 0:
        # fallback: maybe `codex --help` only
        code2, out2, err2 = run_capture(["codex", "--help"])
        help_text += "\n" + out2 + "\n" + err2

    # model/reasoning flags (best-effort)
    caps["has_model"] = bool(re.search(r"--model\b", help_text))
    caps["has_reasoning"] = bool(re.search(r"--reasoning\b", help_text))
    caps["has_skip_git_repo_check"] = bool(re.search(r"--skip-git-repo-check\b", help_text))
    caps["has_sandbox"] = bool(re.search(r"--sandbox\b", help_text))

    # prompt flags
    caps["has_prompt_file"] = bool(re.search(r"--prompt[-_]file\b|--prompt-file\b", help_text))
    caps["has_prompt"] = bool(re.search(r"--prompt\b", help_text))

    # json output flags (varies): record the best supported way
    caps["json_mode"] = None
    if re.search(r"--json\b", help_text):
        caps["json_mode"] = {"type": "flag", "name": "--json"}
    elif re.search(r"--output[-_]format\b", help_text):
        caps["json_mode"] = {"type": "kv", "name": "--output-format", "value": "json"}
    elif re.search(r"--format\b", help_text):
        caps["json_mode"] = {"type": "kv", "name": "--format", "value": "json"}

    # write flags (try to avoid read-only)
    # common variants: --workspace-write, --allow-write, --read-write
    for cand in ["--workspace-write", "--allow-write", "--read-write", "--write"]:
        if cand in help_text:
            caps["write_flag"] = cand
            break

    return caps


def load_prompt_template(kind: str) -> str:
    path = PROMPTS_DIR / f"{kind}.prompt.txt"
    if not path.exists():
        # fallback minimal
        return "{{TASK_BODY}}\n"
    return path.read_text(encoding="utf-8")


def render_prompt(task: Task, policy_lock: str) -> str:
    body = [
        f"TASK_ID: {task.task_id}",
        f"PHASE: {task.phase}",
        f"TYPE: {task.type}",
        f"TITLE: {task.title}",
        "",
        "POLICY_LOCK (must follow):",
        policy_lock.strip(),
        "",
        "TASK DESCRIPTION:",
        task.description.strip(),
        "",
        "ACCEPTANCE CRITERIA:",
        "- " + "\n- ".join(task.acceptance_criteria),
        "",
        "SCOPE LIMITS:",
        json.dumps(task.scope_limits, indent=2, ensure_ascii=False),
        "",
        "VERIFICATION COMMANDS (must run locally):",
        "- " + "\n- ".join(task.verification_commands),
        "",
        "OUTPUTS:",
        "- Update docs/TRACEABILITY.md if you add/change requirements/tasks.",
        "- If you change any policy, add an ADR and create a decision task.",
    ]
    return "\n".join(body) + "\n"


def build_codex_command(caps: Dict[str, Any], prompt_text: str, log_path: Path) -> List[str]:
    cmd = ["codex", "exec"]

    if caps.get("has_skip_git_repo_check"):
        cmd.append("--skip-git-repo-check")

    # write flag if available
    if caps.get("write_flag"):
        cmd.append(str(caps["write_flag"]))
    elif caps.get("has_sandbox"):
        # Newer codex CLI variants expose sandbox mode instead of dedicated write flags.
        cmd += ["--sandbox", "workspace-write"]

    # model/reasoning (best-effort)
    model_name = os.environ.get("CODEX_MODEL", "gpt-5.3-codex").strip()
    if caps.get("has_model"):
        cmd += ["--model", model_name]
    if caps.get("has_reasoning"):
        cmd += ["--reasoning", "high"]

    # output (best-effort)
    jm = caps.get("json_mode")
    if isinstance(jm, dict):
        if jm.get("type") == "flag":
            cmd.append(str(jm.get("name")))
        elif jm.get("type") == "kv":
            cmd += [str(jm.get("name")), str(jm.get("value"))]

    # prompt passing strategy
    # prefer prompt-file if available
    if caps.get("has_prompt_file"):
        prompt_file = log_path.with_suffix(".prompt.txt")
        prompt_file.write_text(prompt_text, encoding="utf-8")
        cmd += ["--prompt-file", str(prompt_file)]
    elif caps.get("has_prompt"):
        cmd += ["--prompt", prompt_text]
    else:
        # positional fallback
        cmd.append(prompt_text)

    return cmd


def run_verification(cmds: List[str]) -> None:
    # Use bash -lc so that scripts/env.sh can be sourced by commands.sh
    for c in cmds:
        p = subprocess.run(["bash", "-lc", c], cwd=str(ROOT_DIR))
        if p.returncode != 0:
            raise RuntimeError(f"Verification failed: {c} (exit {p.returncode})")


def parse_tasks(data: Dict[str, Any]) -> Tuple[str, List[Task]]:
    policy_lock = data.get("policy_lock") or data.get("policy_lock_text", "")
    tasks_raw = data.get("tasks", [])
    tasks: List[Task] = []
    for t in tasks_raw:
        tasks.append(
            Task(
                task_id=t["task_id"],
                phase=t["phase"],
                type=t["type"],
                title=t["title"],
                description=t.get("description", ""),
                acceptance_criteria=t.get("acceptance_criteria", []),
                estimated_minutes=int(t.get("estimated_minutes", 60)),
                scope_limits=t.get("scope_limits", {}),
                verification_commands=t.get("verification_commands", []),
                stop_after=bool(t.get("stop_after", False)),
            )
        )
    return policy_lock, tasks


def main() -> int:
    if not TASKS_PATH.exists():
        print(f"ERROR: tasks file not found: {TASKS_PATH}", file=sys.stderr)
        return 1
    if which("codex") is None:
        print("ERROR: 'codex' CLI not found in PATH", file=sys.stderr)
        return 1

    data = load_json(TASKS_PATH)
    policy_lock_text, tasks = parse_tasks(data)

    state = load_state()
    completed = set(state.get("completed", []))

    caps = detect_codex_exec_capabilities()

    # Run tasks in order
    for task in tasks:
        if task.task_id in completed:
            continue

        # Select prompt template by type
        kind = task.type
        template = load_prompt_template(kind)
        task_prompt = render_prompt(task, policy_lock_text)
        prompt_text = template.replace("{{TASK_BODY}}", task_prompt)

        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"{task.task_id}-{ts}.log"

        cmd = build_codex_command(caps, prompt_text, log_path)

        print(f"==> Running task {task.task_id}: {task.title}")
        print(f"    codex cmd: {' '.join(cmd[:6])} ...")

        code, out, err = run_capture(cmd, cwd=ROOT_DIR)
        log_path.write_text(out + "\n\n[STDERR]\n" + err, encoding="utf-8")

        # If codex failed, stop; user can inspect log and resume.
        if code != 0:
            print(f"ERROR: codex exec failed (exit {code}) for task {task.task_id}", file=sys.stderr)
            print(f"See log: {log_path}", file=sys.stderr)
            return code

        # Run verification commands
        try:
            run_verification(task.verification_commands)
        except Exception as e:
            print(f"ERROR: verification failed for task {task.task_id}: {e}", file=sys.stderr)
            print(f"See log: {log_path}", file=sys.stderr)
            return 1

        # Mark complete
        state.setdefault("completed", []).append(task.task_id)
        state.setdefault("history", []).append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "type": task.type,
                "phase": task.phase,
                "timestamp": ts,
                "log": str(log_path),
            }
        )
        save_state(state)

        # Stop if requested
        if task.stop_after or task.type in ("decision", "checkpoint"):
            print(f"STOP_AFTER task: {task.task_id} ({task.type}). Exiting with {STOP_CODE}.")
            return STOP_CODE

    print("All tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
