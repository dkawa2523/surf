# EVAL_PROTOCOL — 検証ゲート（P0で必ず通す）

この文書は「P0完了判定」を自動化するための検証プロトコルです。
すべて `scripts/commands.sh` をSSOTとします。

---

## P0 ゲート（Must）

### Gate 1: preflight
- 目的：python/pip/importパスを確定し、以降のコマンド事故を防ぐ
- コマンド：
```bash
./scripts/preflight.sh
```
- 合格条件：
  - `scripts/env.sh` が生成される
  - 次のGate2が実行可能

### Gate 2: verify（import + unit-like + in-memory smoke）
- 目的：最低限のimport、最小単体検証、合成データでのインメモリスモーク実行
- コマンド（SSOT）：
```bash
./scripts/commands.sh verify -- --quick
```
- 合格条件（例）：
  - `./scripts/commands.sh verify -- --quick` が 0 で終了
  - required check がすべて `ok:` で終了し、optional 依存不足は `skip(optional)` として扱われる

### Gate 3: cli help（入口が壊れていない）
- コマンド（SSOT）：
```bash
./scripts/commands.sh run -- --help
```
- 合格条件：ヘルプが表示され、0で終了

---

## P1以降（参考）
- 学習の再現性：seed固定、同一configで近い結果になること
- 評価指標：Vn誤差、SDF誤差、断面プロファイル誤差、特徴量誤差など複数
- OOD検知：条件空間/形状埋め込み距離でin-domain判定
- 実測吸収：latent同定（MAP→SBI）でSEM特徴への整合
