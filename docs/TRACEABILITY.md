# TRACEABILITY — 要件 → タスク対応表（100%）

**ルール**：`docs/REQUIREMENTS.md` の全要件は、必ず `tasks/tasks.json` の1つ以上の `task_id` に紐づく。

更新履歴:
- 2026-02-12: `DECISION-P1-001` に合わせて依存タスクへ `depends_on` を付与（要件IDの対応関係は変更なし）。
- 2026-02-14: Pipeline再編要件 (`R-MUST-019`〜`R-MUST-024`) と P1追加タスク (`DECISION-P1-002`, `P1-005`〜`P1-014`, `DECISION-P1-003`) を追記。
- 2026-02-14: `DECISION-P1-002` に合わせて config/loader 連携タスク (`P2-006`, `P2-007`) へ `depends_on` を追記。
- 2026-02-14: `DECISION-P1-002`/`DECISION-P1-003` を Accepted として ADR-0003/0004 に確定反映。
- 2026-02-14: Accuracy-first再編実装反映（core共通化、manifest拡張、verify分割、engine抽象、stage external_inputs拡張）。
- 2026-02-14: I/F粒度同期（OOD二層スコア、feature contract、3D synthetic、optimize履歴、指標命名）を要件/対応表へ反映。
- 2026-02-14: 未決仕様のdecision分割（単位/Δt、材料ラベル、SEM観測形態）を追加し、P1依存を明示化。
- 2026-02-14: DECISION-P1-004/005/006 のADR本文（0005/0006/0007）を追加。
- 2026-02-14: DECISION-P1-004/005/006 を Option A で Accepted 化し、GAPS を解消済みに更新。
- 2026-02-14: optional依存 quick/full 検証契約、observation/prior plugin 化、deprecated 導線を反映。
- 2026-02-14: 生成物衛生化（`src/**/__pycache__`, `*.pyc`, `src/wafer_surrogate.egg-info`）を削除し `.gitignore` を更新。
- 2026-02-14: inference mode resolver backend 抽象と verify runtime capability 表示を追加。
- 2026-02-14: legacy線形 sparse 学生/教師実装を `models/api.py` へ集約し、未参照 `models/sparse_student.py` / `models/sparse_teacher.py` を削除。
- 2026-02-14: 未参照 `workflows/distillation.py` を削除し、蒸留導線を pipeline train (`mode=sparse_distill`) へ一本化。
- 2026-02-14: `data/h5_dataset.py` の deprecated API (`NarrowBandSample`, `dataset_to_samples`, `samples_to_dense_rows`) を削除し、`NarrowBandDatasetReader`/`PointSampler` へ統一。
- 2026-02-14: verify 二段運用を確定（`--quick` は ME shim 許可、`--full` は実MinkowskiEngine必須）し、runtime capability に `MinkowskiEngine_real/shim` 判定を追加。
- 2026-02-14: `verify -- --full` と `verify-full` の挙動を統一し、`WAFER_SURROGATE_REQUIRE_REAL_ME=1` で full 時に shim を自動無効化する実装へ更新。
- 2026-02-15: macOS向け optional-full セットアップ手順を追加（`torch==2.2.2` 固定 + MEビルドスクリプト `scripts/install_optional_full_macos.sh`）。
- 2026-02-15: `torchmd-net-cpu` 競合整理スクリプト `scripts/manage_torchmd_env.sh` を追加し、削除/別環境分離の運用手順を README へ追記。
- 2026-02-15: `scripts/commands.sh` に `torchmd` サブコマンドを追加し、依存競合整理を SSOT 経由で実行できるよう統一。
- 2026-02-15: `manage_torchmd_env.sh` に `--skip-install` を追加し、外部torchmdビルド失敗時でも「環境分離→必要なら削除」を確実に実行できる運用へ改善。
- 2026-02-15: C/H/M残件実装を反映（`example_accuracy.toml`, `template_run_id`, sparse streaming loader, strict contract/split hard fail, BoTorch plugin, multiview/gaussian_mixture plugin, sparse model profile, cleanup/verify hygiene check）。
- 2026-02-15: 精度強化リファクタを反映（narrow-band契約固定、`preprocess_report.json` + `feature_contract_hash`、sparse学習の early-stop/clip/scheduler、`train_ood_reference.json` と推論閾値再利用、split_info必須キー検証）。
- 2026-02-15: 追加同期（feature contract mismatch文言統一、preprocess bundle/report必須キー検証、leaderboard優先キーメタ保存、split_info正規化、accuracy設定の前処理推奨値固定）。
- 2026-02-15: 残件C/H実装を反映（tabularでも`feature_contract.json`必須化、列順不一致hard fail、`strict_split=true`既定化、OOD report `summary` 追加、leaderboard `ranking_priority` 検証追加）。

| Requirement ID | Task IDs |
|---|---|
| R-MUST-001 | P0-004, P0-005, P0-006, P1-003, P1-004 |
| R-MUST-002 | P0-004, P0-005 |
| R-MUST-003 | P0-002, P0-010, P2-005 |
| R-MUST-004 | DECISION-P1-001, P0-005, P1-004 |
| R-MUST-005 | P1-004, P2-004 |
| R-MUST-006 | P0-002, P0-006, P1-005, P1-006, P1-007, P1-008, P1-009, P1-010 |
| R-MUST-007 | P0-002, P0-010, P1-005, P1-010, P1-011 |
| R-MUST-008 | P0-007, P0-006, P0-010, P1-009, P1-010, P1-011 |
| R-MUST-009 | P0-007, P1-010 |
| R-MUST-010 | P0-006, P0-010, P1-010 |
| R-MUST-011 | P0-010, P1-010, P2-005 |
| R-MUST-012 | P0-001, P0-006, P1-005 |
| R-MUST-013 | P0-001, P0-009, P1-012 |
| R-MUST-014 | P0-001, P0-009 |
| R-MUST-015 | P0-009, P0-CHECKPOINT |
| R-MUST-016 | P0-009, P0-CHECKPOINT |
| R-MUST-017 | P0-CHECKPOINT |
| R-MUST-018 | DECISION-P1-001, DECISION-P1-002, DECISION-P1-003, DECISION-P1-004, DECISION-P1-005, DECISION-P1-006, P0-009 |
| R-MUST-019 | P1-005, P1-006, P1-007, P1-008, P1-009, P1-010 |
| R-MUST-020 | P1-005, P1-010, P1-012 |
| R-MUST-021 | P1-005 |
| R-MUST-022 | P1-010 |
| R-MUST-023 | P1-011 |
| R-MUST-024 | P1-005, P1-013, P1-014 |
| R-MUST-025 | P1-007, P1-009, P1-010, P1-013 |
| R-MUST-026 | P1-010, P1-013 |
| R-MUST-027 | P1-005, P1-007, P1-010, P1-013 |
| R-MUST-028 | P1-010, P1-011 |
| R-MUST-029 | P1-009, P1-010, P1-013 |
| R-MUST-030 | P1-013, P1-014 |
| R-MUST-031 | P1-010, P2-006, P2-007 |
| R-SHOULD-001 | P1-002 |
| R-SHOULD-002 | P1-003 |
| R-SHOULD-003 | P0-004, P0-005, P1-003 |
| R-SHOULD-004 | P2-001 |
| R-SHOULD-005 | P2-002 |
| R-SHOULD-006 | DECISION-P1-001, P2-005 |
| R-SHOULD-007 | P0-002, P0-006, DECISION-P1-001 |
| R-SHOULD-008 | P1-004 |
| R-SHOULD-009 | P0-006, P0-010 |
| R-COULD-001 | P2-004 |
| R-COULD-002 | P2-005 |
| R-COULD-003 | P2-003 |
| R-COULD-004 | P2-006 |
| R-COULD-005 | P2-007 |

# VIZ_TRACEABILITY_ADDON — 追加要件 → 追加タスク対応表

このファイルは `docs/TRACEABILITY.md` へ追記するための対応表（提案）です。

| Requirement ID | Task IDs |
|---|---|
| R-SHOULD-VIZ-001 | P0-VIZ-001 |
| R-SHOULD-VIZ-002 | P0-VIZ-002 |
| R-SHOULD-VIZ-003 | P0-VIZ-003 |
| R-SHOULD-VIZ-004 | P0-VIZ-004 |
| R-SHOULD-VIZ-005 | P0-VIZ-001, P0-VIZ-002, P0-VIZ-003, P0-VIZ-004 |
