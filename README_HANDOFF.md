# Codex CLI 自動実装 Handoff Pack（ウェハ加工形状サロゲート）

このパックは、`codex` CLI（モデル: **gpt-5.3-codex**, `reasoning=high`）で、
**自動実行 →（大きい節目だけ）checkpoint停止 → 続きから再開(resume)** を破綻なく回すための Handoff Pack です。

- 実装対象：半導体製造のウェハ加工形状（エッチ/デポ/スパッタ/反射/表面拡散等を含む）MC形状シミュレーションの代替サロゲート
- 重点：時間発展出力、形状連続性、少データ、実測SEM吸収（潜在同定）、原因特定、レシピ探索

---

## 1) 展開
```bash
unzip codex_handoff_pack.zip -d <WORKDIR>
cd <WORKDIR>
```

## 2) 自動実行（P0 → checkpoint停止）
```bash
./scripts/run.sh
```

`run.sh` は以下を行います。
- `MPLCONFIGDIR=/tmp` を標準設定（Matplotlibの環境差事故を抑制）
- `runs/` を作成（生成物・状態ファイル格納）
- `./scripts/preflight.sh` で Python/import/pip を確定し `scripts/env.sh` に書き出し
- `./scripts/codex_autorun.py` で `tasks/tasks.json` を順に実行

## 3) 停止条件（exit code 42）
次のタスク種別で **exit code 42** により停止します。
- `type=decision`：方針未確定のため停止（`docs/GAPS.md` とタスク本文を読んで決定）
- `type=checkpoint`：大きい節目で停止（原則 P0 末尾のみ）

停止したら、指示されたドキュメント（例：`docs/adr/...`）を更新し、再度 `./scripts/run.sh` を実行してください。

## 4) 再開（resume）
再度同じコマンドでOKです。完了済みタスクはスキップされ、続きから再開します。
```bash
./scripts/run.sh
```

状態は `runs/autorun_state.json` に保存されます。

---

## runs/ ディレクトリ運用
- `runs/` は **git管理しません**（`.gitignore` 済み）
- 自動実行のログや生成物、チェックポイント状態を格納します

---

## 依存関係（重要）
`POLICY_LOCK` により、**P0 では依存追加を行いません**。
- P0 は「動くスキャフォールド＋最小スモーク検証」まで
- Torch / MinkowskiEngine / sbi / BoTorch などの追加は、P1で `decision` タスクにより明示的に有効化します

---

## 実行コマンド（Single Source of Truth）
すべて `scripts/commands.sh` に集約されます。
- 実行：`./scripts/commands.sh run`
- 検証：`./scripts/commands.sh verify`

### Pipeline 実行（P1）
新しい stage ベース実行は次で呼び出せます。
```bash
./scripts/commands.sh run -- pipeline run --config configs/example.toml
./scripts/commands.sh run -- pipeline run --config configs/example.toml --stages cleaning,featurization,preprocessing,train,inference
```

---

## トラブルシュート
- `codex` が見つからない：`codex` CLI をインストールし PATH を通してください
- Pythonが見つからない：`python3` か `python` を用意してください（`preflight.sh` が自動検出）
- importが失敗する：`preflight.sh` が `pip install -e .` を試し、不可なら `PYTHONPATH=src` を設定します
