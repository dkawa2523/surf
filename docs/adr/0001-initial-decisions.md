# ADR-0001: Initial decisions & POLICY_LOCK

Date: 2026-02-11

## Context
本プロジェクトは Codex CLI 自動実行（autorun）により、段階的に実装を進める。
python/pip/import、run/test コマンドの揺れ、生成物管理、checkpoint運用を固定しないと破綻する。

## Decision (POLICY_LOCK)
1) **Python実行コマンド**: `preflight.sh` が `python3` を優先して自動検出し、確定値を `scripts/env.sh` に書き出す。
2) **パッケージ実行方式**: srcレイアウトを前提とし、`preflight.sh` が
   - `pip install -e .` を試み、不可なら `PYTHONPATH=src` を設定する（fallback）。
3) **単一の正 (SSOT)**: 実行/検証コマンドは `scripts/commands.sh` に集約する。
4) **依存追加方針**: P0では依存追加を行わない。Torch/MinkowskiEngine/sbi/BoTorch等の追加は P1 の `decision` タスクで明示的に有効化する。
5) **出力ディレクトリ**: 生成物は `runs/` に保存し、git管理しない（.gitignore）。
6) **checkpoint方針**: 原則 P0 末尾の 1 回のみ。
7) **codex exec 実行ポリシー**: autorunは `codex exec --help` で対応フラグを検出し、必ず workspace-write 相当で実行する（read-only事故を防止）。
8) **Matplotlib安定化**: `MPLCONFIGDIR=/tmp` を `run.sh` で標準設定する。

## Consequences
- P0は「依存ゼロで動くスキャフォールド + スモーク検証」まで。
- 追加依存は明示的な決定プロセスを経るため、pip失敗やGPU依存による自動実行停止を避けられる。

## Alternatives considered
- YAML/Hydra を最初から導入：依存追加が必要なため P0では見送る。
- pytest を必須化：環境差があるため P0では `./scripts/commands.sh verify -- --quick` を採用。
