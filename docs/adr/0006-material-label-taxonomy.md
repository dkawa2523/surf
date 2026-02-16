# ADR-0006: Material label taxonomy and encoding policy

Date: 2026-02-14
Status: Accepted

## Context
- `DECISION-P1-005` では、材料ラベル体系を固定しないと
  - feature engineering（材料依存特徴）
  - train/inference の入力契約
  - 将来の多材料評価指標
  が不整合になる。
- 現状は synthetic 主体で材料情報が薄く、後付け統合時に free-text ラベルの混在リスクがある。

## Blocking decisions for DECISION-P1-005
1) 材料ラベルを固定列挙（enum）にするか  
2) エンコードを整数IDにするか one-hot にするか  
3) 未知ラベル（unknown/new material）の扱いをどうするか

## SAFE options (A/B/C)
### A) Fixed enum + integer ID canonical (Recommended)
- 材料ラベルを固定 enum で管理し、内部は整数IDで扱う。
- 推論時 unknown は `material_unknown` へフォールバックし warning を出す。
- Pros: 実装・保存形式が安定し、後方互換を維持しやすい。
- Cons: 新材料追加時に enum 更新が必要。

### B) Fixed enum + one-hot canonical
- 入力契約を one-hot に固定してモデルへ直接投入する。
- Pros: 線形/NNモデルで扱いやすい。
- Cons: 次元増加と schema 管理コストが高い。

### C) Free-text label pass-through
- 材料を文字列のまま受け渡し、モデル前で都度処理する。
- Pros: 追加時の初動は速い。
- Cons: ラベル揺れ・誤記による再現性劣化が大きい。

## Decision (explicit)
- `DECISION-P1-005` は **Option A** を正式採用する。
- 固定案:
  - canonical labels:
    - `material_unknown = 0`
    - `material_si = 1`
    - `material_sio2 = 2`
    - `material_sin = 3`
    - `material_pr = 4`
  - artifact 契約:
    - `material_schema.version`
    - `material_schema.labels`（name->id）
    - `material_schema.unknown_policy`（`map_to_unknown`）
  - 推論:
    - 未知ラベルは `material_unknown` へ写像し、`manifest.warnings` に記録

## Task gating
- `P1-007`, `P1-009`, `P1-010` は本decision確定前に材料ラベル仕様をハードコードしない。
- 材料列が必要な機能は `DECISION-P1-005` の最終決定後に反映する。

## Consequences
- 材料ラベルの揺れを抑制し、学習再現性を確保できる。
- 将来の材料拡張も version 管理で安全に行える。

## Confirmation note
- 2026-02-14: ユーザ承認により Option A を確定。
