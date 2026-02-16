# ADR-0004: Optimization engine policy (built-in standard + optional external optimizers)

Date: 2026-02-14
Status: Accepted

## Context
- `Inference.optimize` は将来的に BO/SBI を含む高度化が想定される一方、`POLICY_LOCK` と autorun 安定性の観点で依存追加は慎重に扱う必要がある。
- `DECISION-P1-003` では、最適化エンジンを内製機能に限定するか、外部最適化器を optional で受け入れるかを確定する必要がある。

## SAFE options (A/B/C)
### A) Built-in only
- 内製 random/grid のみを正式サポートし、外部最適化器を導入しない。
- Pros: 依存・運用が最小。
- Cons: 探索効率と拡張性が限定される。

### B) Built-in standard + optional external optimizer (Selected)
- 標準経路を Optuna ベースで定義しつつ、外部最適化器は optional plugin として許可する。
- 依存不足時は fail fast ではなく、random/grid へ明示フォールバックする。
- Pros: 比較可能な標準探索を維持しながら、目的別の高度化を追加できる。
- Cons: エンジン選択と再現条件（seed/sampler/log形式）の管理が必要。

### C) External optimizer required
- Optuna/BoTorch/SBI など外部最適化器を必須依存として固定する。
- Pros: 高度機能を最短で利用しやすい。
- Cons: 環境依存で verify/autorun の失敗率が上がる。

## Decision (explicit)
`DECISION-P1-003` は **Option B** を正式採用する。

- 標準方針:
  - optimize の標準インタフェースは Optuna ベースで定義する。
  - ただし実行環境で optional 依存が無い場合でも、内製 random/grid で実行継続できることを必須とする。
- 外部最適化器:
  - SBI/BO 高度化は plugin として接続可能にする。
  - 導入時は optional import + graceful degrade を維持し、依存未導入時に `verify --quick` を壊さない。
- 再現性:
  - sampler 名、seed、試行回数、目的指標、最良解を成果物に記録する。
  - 既定の結果出力は `runs/<run_id>/inference/outputs/` と `runs/<run_id>/leaderboard/` に統一する。

## Consequences
- 実装は `inference.optimize` にエンジン抽象を持ち、`optuna` と `random/grid` を切り替え可能にする。
- P2 の SBI/BO タスクは optional plugin 前提で実装できる。
- 依存の有無に関わらず、最小ワークフローは継続実行できる。

## Confirmation note
- 2026-02-14: ユーザ承認済み（「B 内製標準 + 外部最適化器optional」「標準はOptunaベース」を選択）。
