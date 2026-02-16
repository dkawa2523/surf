# ADR-0007: SEM observation modality contract

Date: 2026-02-14
Status: Accepted

## Context
- `DECISION-P1-006` では、SEM観測の標準形態（断面/トップダウン/複数視点）を固定する必要がある。
- 観測形態が未確定のままだと
  - `observation` 特徴抽出
  - `inference.calibration`（MAP/SBI）
  - 評価可視化（compare sections / residual）
  のI/O契約が揺らぐ。

## Blocking decisions for DECISION-P1-006
1) canonical modality を cross-section / top-view / multi-view のどれにするか  
2) 最低限必須の観測特徴セットを何にするか  
3) modality を増やすときの後方互換ルールをどうするか

## SAFE options (A/B/C)
### A) Cross-section canonical + extensible multi-view (Recommended)
- まず断面（XZ）を canonical とし、top-view/multi-view は optional 拡張として扱う。
- 既存 compare-sections と整合し、段階導入しやすい。
- Pros: 現行ワークフローに自然接続できる。
- Cons: 初期は top-view 主体データの活用が限定される。

### B) Top-view canonical
- 平面観測を標準とし、断面は補助扱いにする。
- Pros: 一部運用ではデータ取得が容易。
- Cons: 側壁情報や深さ方向形状の識別力が不足しやすい。

### C) Multi-view required from start
- 複数視点統合を初期から必須化する。
- Pros: 観測情報量は最大。
- Cons: データ要件・実装複雑性が高く、運用開始が遅れる。

## Decision (explicit)
- `DECISION-P1-006` は **Option A** を正式採用する。
- 固定案:
  - canonical modality: `cross_section_xz`
  - required feature schema:
    - `cd_top`, `cd_mid`, `cd_bottom`
    - `sidewall_angle`
    - `curvature_proxy`
    - `footing_proxy`
  - optional extension:
    - `top_view_*`
    - `multi_view/<view_id>/*`
  - artifact metadata:
    - `observation.modality`
    - `observation.schema_version`
    - `observation.feature_names`

## Task gating
- `P1-010`（inference/calibration）は本decision確定前に modality 固定値を増やさない。
- 観測特徴の追加は schema version を上げ、既存キー互換を維持する。

## Consequences
- calibration/eval で最低限の比較軸を固定できる。
- multi-view 拡張は breaking change なしで追加できる。

## Confirmation note
- 2026-02-14: ユーザ承認により Option A を確定。
