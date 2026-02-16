# ADR-0003: YAML adoption policy and config compatibility

Date: 2026-02-14
Status: Accepted

## Context
- `POLICY_LOCK` により、P0 は依存追加禁止で進行し、設定は TOML 中心で運用してきた。
- 現在、`src/wafer_surrogate/config/loader.py` は TOML + optional YAML (`pyyaml`) を扱える一方、
  既存 CLI の一部は TOML 前提のまま。
- P1 以降の pipeline 展開では設定互換ルールが未確定だと、移行方針と検証条件を固定できない。

## Blocking decisions for DECISION-P1-002
1) YAML parser を何にするか  
2) YAML を optional dependency にするか、必須化するか  
3) TOML 互換をどこまで維持するか  
4) 移行ルール（YAML-first の適用時期とゲート条件）

## SAFE options (A/B/C)
### A) TOML-only (Conservative)
- YAML を採用しない。設定は TOML のみ。
- Pros: 依存ゼロを維持し、verify の安定性が高い。
- Cons: YAML-first を期待する運用と乖離し、将来移行コストが先送りになる。

### B) Dual-format staged rollout (Recommended)
- TOML を baseline として維持しつつ、YAML は optional (`pyyaml`) で段階導入する。
- YAML 依存が無い環境では明示エラーで fail fast（自動 install はしない）。
- Pros: 既存互換を壊さず、YAML 導入を前進できる。
- Cons: 当面は TOML/YAML 併存に伴う運用ルール管理が必要。

### C) YAML-first required (Aggressive)
- YAML を標準化し、`pyyaml` を必須依存にする。TOML は移行期間限定で互換提供。
- Pros: 設定表現を一本化しやすい。
- Cons: 依存追加により環境失敗率が上がり、minimal 環境の verify 安定性が下がる。

## Decision (explicit)
`DECISION-P1-002` は **Option B (Dual-format staged rollout)** を正式採用する。

- Parser choice:
  - TOML: `tomllib`（必要に応じて `tomli` fallback）
  - YAML: `pyyaml`（optional）
- Optional dependency behavior:
  - YAML ファイル指定時に `pyyaml` が無い場合は明示エラーで停止する（silent fallback しない）。
  - `verify --quick` は引き続き追加依存なしで通ることを条件とする。
- TOML compatibility:
  - P1 期間は TOML を baseline として必須互換維持。
  - 既存 `configs/example.toml` は SSOT として維持。
- Migration rule:
  - P1 では一括変換を行わない（mass_refactor 回避）。
  - YAML-first へ昇格する場合は、別 decision で
    - 依存扱い（required 化の可否）
    - verify 基準
    - 既存 CLI の互換期限
    を確定してから実施する。

## Task gating
- YAML-first 前提を取り得るタスクは `DECISION-P1-002` 完了を前提にする。
- config/loader に関与する下流タスクは本決定を依存条件として扱う。

## Consequences
- P1 実装は TOML 互換を維持したまま進行できる。
- YAML 利用は optional で開始し、環境差による autorun 停止リスクを抑える。
- YAML-first の本格移行は、次の decision で明示的に確定する。

## Confirmation note
- 2026-02-14: ユーザ承認済み（「TOML+YAML併用（推奨）」を選択）。
