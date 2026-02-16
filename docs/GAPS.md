# GAPS — 未決事項 / 不足情報

この文書は「勝手に仕様を追加しない」ためのストッパーです。
未決事項は `tasks/tasks.json` の `type=decision` に紐づけて、そこで停止します。

---

## DECISION-P1-001 で整理した項目（解消済み）
参照: `docs/adr/0002-constraints.md`

### G-001 REPO_INFO（既存repoの有無）
- 状態: 解消済み（Option B / Accepted）
- 決定方針: **既存repo継続**
- ブロック理由: 新規repo方針だと path/CI/run導線が全面変更となり、P1/P2実装開始条件が確定しない。
- 対応タスク: DECISION-P1-001

### G-002 CONSTRAINTS（network/pip/GPU/依存追加）
- 状態: 解消済み（Option B / Accepted）
- 決定方針:
  - network/pip はデフォルト禁止、必要時のみ decision で限定解放
  - 依存追加は P1/P2 で条件付き許可（optional import + graceful degrade）
  - GPU は任意（必須要件にしない）
- ブロック理由: 許可境界が未確定だと依存導入タスクの受入条件を定義できない。
- 対応タスク: DECISION-P1-001

## G-003 データ仕様の一部未確定
- 状態: 解消済み（`docs/adr/0005-unit-and-dt-semantics.md`, `docs/adr/0006-material-label-taxonomy.md`）
- 決定内容:
  - 単位/Δt: Option A（voxel-step canonical + optional physical conversion）
  - 材料ラベル: Option A（fixed enum + integer ID canonical）
- 影響：SDF符号、教師Vn生成、評価メトリクス契約を固定。
- 対応タスク：
  - 単位/Δt: DECISION-P1-004
  - 材料ラベル: DECISION-P1-005

## G-004 実測SEMの形態（断面/トップダウン/複数視点）
- 状態: 解消済み（`docs/adr/0007-sem-observation-modality-contract.md`）
- 決定内容:
  - Option A（cross-section canonical + extensible multi-view）
- 形状特徴量の設計や観測モデル契約を固定。
- 対応タスク：DECISION-P1-006

## G-005 設定形式（YAML移行ポリシー）
- 状態: 解消済み（`docs/adr/0003-yaml-adoption-policy.md`）
- 決定内容:
  - Option B（TOML baseline + YAML optional）を正式採用
  - YAML 指定時に `pyyaml` が無い場合は明示エラー
  - `verify --quick` は依存未追加でも成立
- 影響：設定ローダー、CLI互換、verify基準、autorun実行安定性。
- 対応タスク：DECISION-P1-002

## G-006 最適化エンジン方針
- 状態: 解消済み（`docs/adr/0004-optimization-engine-policy.md`）
- 決定内容:
  - Option B（内製標準 + 外部最適化器 optional）を正式採用
  - optimize 標準I/Fは Optuna ベース
  - 依存未導入時は random/grid にフォールバックして継続
- 影響：依存追加、再現性、比較指標の標準化。
- 対応タスク：DECISION-P1-003
