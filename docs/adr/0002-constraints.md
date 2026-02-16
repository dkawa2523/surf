# ADR-0002: Constraints and dependency policy for P1/P2

Date: 2026-02-12
Status: Accepted

## Context
- P1/P2 では `h5py/zarr` や Torch 系（例: sbi/BoTorch を含む）など、追加依存が必要になる可能性がある。
- `POLICY_LOCK` により P0 では依存追加は禁止、P1 の decision で有効化判断を行う必要がある。
- 未決のまま実装を進めると、ネットワーク不可・GPU不可・pip不可の環境でタスク失敗し、autorun が停止する。

## Blocking decisions
1) 既存repo継続か、新規repo分離か  
2) network/pip を許可するか  
3) 依存追加をどこまで許可するか  
4) GPU を必須にするか（または任意にするか）

## Options (SAFE A/B/C)
### A) Conservative (最小リスク)
- 既存repo継続
- network/pip 不許可
- 依存追加は引き続き禁止
- GPUは使っても良いが要件はCPUのみ
- Pros: 再現性が高く、環境差で止まりにくい
- Cons: P2 の高度機能（SBI/BO/MFBO等）は実装範囲が大きく制限される

### B) Balanced staged rollout (Recommended)
- 既存repo継続
- network/pip は「タスク単位で明示許可時のみ」許可
- 依存追加は P1 以降で許可。ただし optional import + graceful degrade を維持
- GPUは任意（必須にしない）。検証ゲートはCPUで通ることを条件にする
- Pros: 安全性を保ちつつ、必要な依存導入で P1/P2 を前進できる
- Cons: 依存導入のたびに判断・記録（ADR/decision）が必要

### C) Aggressive enablement
- 新規repo分離または大幅再構成を許可
- network/pip 常時許可
- 依存追加を広く許可
- GPU前提で実装
- Pros: 最新スタックで高速に機能実装しやすい
- Cons: 環境依存・運用コスト・失敗率が上がり、POLICY_LOCK運用と衝突しやすい

## Decision
- **Option B** を正式採用する。
- 具体化:
  - Repo: 既存repo継続（new repo は採用しない）
  - Network/pip: デフォルト禁止。必要時のみ decision で限定解放
  - Dependency add: P1/P2 で条件付き許可（optional/skip可能な実装を維持）
  - GPU: 非必須（利用可能なら任意で活用）

## Consequences
- 依存を要するタスクは `DECISION-P1-001` を前提条件として明示する。
- 実装タスクは「依存なし代替経路」または「依存不足時のスキップ」を必須設計とする。
- `scripts/commands.sh verify` は引き続きCPU環境で通過可能に維持する。
- 検証モードは `verify --quick`（依存なし基準）と `verify --full`（optional依存必須）を分離する。
  - `verify --quick`: MinkowskiEngine shim を許可
  - `verify --full`: 実MinkowskiEngine必須（shim は失敗扱い）

## Confirmation note
- 2026-02-14: ユーザ承認により Option B を確定。
- 2026-02-14: 運用詳細を Option D（二段検証: quickはshim可 / fullは実ME必須）として確定。
- C を選ぶ場合は POLICY_LOCK 変更のため、別 decision + ADR 追加が必要。
