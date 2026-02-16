# ADR-0005: Unit system and dt semantics for SDF/Vn pipeline

Date: 2026-02-14
Status: Accepted

## Context
- `DECISION-P1-004` では、SDF/level-set 学習で使う単位系（nm/voxel）と `dt` の意味を固定する必要がある。
- 未確定のままだと、`Featurization -> Train -> Inference` で
  - `vn_target` の次元
  - rollout 更新量（`phi <- phi - dt * vn * |grad phi|`）
  - 評価指標の比較可能性
  が環境・データごとに変わる。
- 現状コードは synthetic 生成中心で voxel-index 基準の演算が多く、実測連携時は変換契約が不足している。

## Blocking decisions for DECISION-P1-004
1) 内部計算の基準単位を voxel 基準にするか、物理単位（nm）基準にするか  
2) `dt` を「物理時間」として扱うか、「進行ステップ係数」として扱うか  
3) `vn_target` と metrics の出力単位を何で標準化するか

## SAFE options (A/B/C)
### A) Internal voxel-step canonical + optional physical conversion (Recommended)
- 学習/推論の内部契約は `voxel` と `step` を正規化単位にする。
- 物理単位（nm, s）は optional metadata (`nm_per_voxel`, `dt_seconds`) で保持し、必要時のみ変換。
- Pros: 既存実装との整合が高く、依存追加なしで再現性を保ちやすい。
- Cons: 物理解釈は明示変換が必要。

### B) Physical-unit canonical (nm/s)
- 内部契約を物理単位に統一し、全入力で単位変換を必須化する。
- Pros: 実測・装置条件との解釈が直感的。
- Cons: データ前処理負荷が高く、単位欠損時に実行不能になりやすい。

### C) Dataset-local free unit
- データごとに単位を任意運用し、コード側では強制しない。
- Pros: 初期導入は簡単。
- Cons: 学習比較・再現性が崩れ、評価の整合が取れない。

## Decision (explicit)
- `DECISION-P1-004` は **Option A** を正式採用する。
- 固定案:
  - 内部単位: `phi` は voxel-index 基準、`dt` は step 係数、`vn_target` は voxel/step。
  - optional metadata:
    - `units.nm_per_voxel`（float, optional）
    - `units.dt_seconds`（float, optional）
    - `units.semantic`（`voxel_step` or `physical_time`）
  - 出力指標:
    - 必須: voxel-step 基準（`vn_mae`, `vn_rmse`）
    - optional: metadata が揃う場合に nm/s 換算指標を併記

## Task gating
- `P1-007`, `P1-008`, `P1-009`, `P1-010` は本decision確定前に単位仕様を追加しない。
- 仕様追加が必要な場合は `DECISION-P1-004` 完了後に実装する。

## Consequences
- 内部演算の比較可能性を維持しつつ、将来の物理換算に拡張できる。
- 実測連携時の単位不整合は metadata で明示検知できる。

## Confirmation note
- 2026-02-14: ユーザ承認により Option A を確定。
