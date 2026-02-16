# AGENTS.md — Codex CLI 実装エージェント向けルール

このリポジトリは **Codex CLI の自動実行**を前提にしています。
タスク実行中は、以下のルールを最優先で守ってください。

## 0) 最優先：POLICY_LOCK
- `docs/adr/0001-initial-decisions.md` に記録された **POLICY_LOCK** を破らない。
- 破りたくなったら、勝手に変更せず `docs/adr/` に提案を書き、`type=decision` タスクを追加して停止する。

## 1) 1タスク = 1論点
- `tasks/tasks.json` の scope 制約（max_changed_files / max_diff_lines / allowed_dirs）を厳守。
- 1タスクで複数論点をまとめない。必要ならタスクを分割する。

## 2) 仕様・方針を勝手に追加しない
- 本チャットと `docs/CONTEXT.md` / `docs/REQUIREMENTS.md` に無い前提・仕様は追加しない。
- 未決事項は `docs/GAPS.md` に追記し、`type=decision` タスクを追加して停止する。

## 3) 依存追加・GPU前提・ネットワーク前提は decision で止める
- `forbidden_actions` に該当する行為（例：dependency_add, network_enable, mass_refactor）は行わない。
- 追加したい場合は ADR 提案→ decision タスク→ユーザ決定 の順で進める。

## 4) ドキュメントを「唯一の仕様」として更新
- 重要事項は `docs/adr/` に残す（チャットログに閉じない）。
- 要件の追加/変更は `docs/REQUIREMENTS.md` に反映し、`docs/TRACEABILITY.md` と `tasks/tasks.json` を整合させる。

## 5) 検証の単一の正（SSOT）
- 実行・検証コマンドは `scripts/commands.sh` を参照し、直接ハードコードしない。
- `python` vs `python3` 問題は `scripts/preflight.sh` が解決し、`scripts/env.sh` に固定する。

## 6) チェックポイント運用
- checkpoint は原則 P0 末尾のみ。
- checkpointで止まったら最初からやり直さない。`runs/autorun_state.json` を読み、未完了タスクから再開する。
