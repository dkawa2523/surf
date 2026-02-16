# ARCHITECTURE — パッケージ境界と依存方向

## 0) 最優先
- 実行・検証コマンドのSSOT（Single Source of Truth）は `scripts/commands.sh`。
- Python環境の確定は `scripts/preflight.sh` → `scripts/env.sh`。
- POLICY_LOCK は `docs/adr/0001-initial-decisions.md`。

---

## 1) リポジトリレイアウト（現行）
- **src レイアウト**を採用する（import事故を避けるため `preflight.sh` で調整）。

```
src/
  wafer_surrogate/
    cli.py                # 薄いCLI入口（parser配線のみ）
    cli_parser.py         # CLI parser定義
    cli_commands/         # command handler群（workflow/viz/common）
    core/                 # 共通I/O, rollout, OOD
      feature_contract.py # feature contract / predict_vn 契約検証
    runtime/
      capabilities.py     # optional依存可用性の単一判定点
    pipeline/             # orchestrator + stage実装
    verify/               # 検証ゲート（checks分割）
    config/
      loader.py           # toml/yaml/json 設定ロード
      schema.py           # pipeline run config schema
    data/
      io.py               # Dataset I/O 抽象とアダプタ（HDF5/Zarrなど）
      sem.py              # SEM入力（輪郭/特徴量）
      synthetic.py        # 2D/3D 合成データ
      mc_logs.py          # privileged log（real/proxy）
    features.py           # 特徴抽出registry
    preprocess.py         # 前処理/変換（standard/robust/quantile/pca/log1p）
    geometry.py           # 勾配, levelset update, reinit
    models/
      api.py              # Model Interface（predict_vn 等）
      sparse_unet_film.py # SparseTensor モデル（optional）
    training/
      train_sparse_distill.py # 蒸留学習ループ（optional依存）
    inference/
      simulate.py         # φ更新を回すサロゲートシミュレーター
      ood.py              # 条件空間OOD
      calibrate.py        # latent同定（MAP + optional SBI posterior）
    observation/
      registry.py         # 観測モデルI/Fとregistry
      plugins/            # baseline / 将来レンダラ plugin
    prior/
      registry.py         # prior I/Fとregistry
      plugins/            # gaussian / 将来生成prior plugin
    optimization/
      engines/            # builtin/optuna/plugin 抽象
```

---

## 2) 依存方向（破綻防止）
- `geometry/` と `data/` は **モデルに依存しない**。
- `models/` は `geometry/` に依存してよいが、`training/` には依存しない。
- `training/` は `models/` と `data/` に依存する。
- `inference/` は `models/` と `geometry/` に依存する。
- `observation/` は shape state を受けて観測特徴へ写像し、`inference/calibrate` と `metrics/eval` から利用する。
- `prior/` は latent の `sample/score` 契約のみを提供し、`cli`/`inference` から利用する。
- `optimization/` は `inference/` を呼び出すが、逆依存は禁止。
- optional依存判定（torch/MinkowskiEngine/sbi/botorch/optuna）は `runtime/capabilities.py` 以外で直接行わない。
- 開発/CI の依存なし環境向けに `src/MinkowskiEngine/` 互換 shim を置き、API契約の回帰を維持する。
- 検証ポリシーは二段運用とする:
  - `verify --quick`: shim許可
  - `verify --full`: 実MinkowskiEngine必須（`WAFER_SURROGATE_REQUIRE_REAL_ME=1` でshimを無効化し、shim検出時はfail）

---

## 3) 「用途ごとに独立に実行できる」設計
- 特長量化・前処理・学習・推論は **それぞれ単体で呼べる**。
- さらに、用途に応じてパイプラインとして組み替え可能にする。
  - 例：`features -> preprocess -> train`
  - 例：`preprocess -> infer -> eval`
  - 順序は固定しない（ただし各ステップのI/O契約は固定する）。

---

## 4) 設定方針
- P0では依存追加を避けるため、設定は **TOML** を第一選択とする（Python標準 `tomllib`）。
- 追加要望（Hydra/YAML）は `type=decision` で確定後に導入する。

---

## 5) 実行・検証コマンド（SSOT）
- すべて `scripts/commands.sh` に集約。
- タスク本文・READMEでは `scripts/commands.sh run/verify` を参照し、python直叩きを避ける。
- `verify` は `--quick`（依存なし基準 / shim許可）と `--full`（optional依存必須 / 実ME必須）を使い分ける。

---

## 6) 生成物
- `runs/` 配下に、ログ・モデル・評価・状態（autorun_state.json）を保存。
- `runs/` はgit管理しない（`.gitignore`）。

## 7) 微分可能レンダリングへの拡張点（P2 stub）
- 現状は `observation` plugin registry で `baseline` を提供し、統計特徴に加えて
  CD(top/mid/bottom), sidewall angle proxy, curvature proxy, footing proxy, gradient/narrow-band特徴を含む。
- 将来、同一I/Fで SDF->画像レンダラを差し替えることで、SEM画像空間の損失導入に拡張できる。
- `metrics`/`calibrate` 側は `ObservationModel` 契約に依存するため、plugin差し替えで再利用可能。

## 8) 生成priorへの拡張点（P2 stub）
- 現状は `prior` plugin registry で `gaussian_latent` を提供。
- 契約: `score_latent(z)` は finite float を返す。
- `cli` は `config[prior]` から prior を構築し、学習/推論サマリへ prior 情報を記録する。
- 将来の拡散/scoreモデルは同一I/Fで登録し、`score_latent(z)` をエネルギー/負対数尤度として実装すれば、既存のMAP/SBIフローへ段階的に統合できる。

## 9) Pipeline実行アーキテクチャ（P1）
- `src/wafer_surrogate/pipeline/orchestrator.py` が stage 実行順と manifest 管理を担う。
- stage 実装は `src/wafer_surrogate/pipeline/stages/` に分離する。
  - `cleaning.py`
  - `featurization.py`
  - `preprocessing.py`
  - `train.py`
  - `inference.py`
- stage 出力は `runs/<run_id>/<stage>/{configuration,logs,outputs}` に統一する。
- 実行全体は `runs/<run_id>/manifest.json` に保存し、stage status / artifacts / metrics / leaderboard を追跡可能にする。
- `manifest` の必須系:
  - `schema_version`
  - `stage_dependencies`
  - `stage_inputs`
  - `runtime_env`
  - `seed_info`
  - `split_info`
  - `warnings`
- 比較評価は `runs/<run_id>/leaderboard/{data_path,model_path}` を標準とする。
- leaderboard の標準順位は `student_mae` を第一、`rollout_short_window_error` を第二キーとする（不在時のみ `mae/rmse` へフォールバック）。
- leaderboard可視化は `runs/<run_id>/leaderboard/viz/` に出力し、matplotlibなし環境ではCSV fallbackを保存する。
- 可視化設定は TOML/YAML 併用:
  - 既定: `configs/visualization.default.yaml`（全トグルON）
  - 優先順位: `CLI --viz-config-yaml` > `stage.visualization` > `run.visualization` > default YAML
  - 依存不足時は失敗ではなく `visualization_manifest.json` に skip 理由を残す。
- `featurization -> train` で feature contract を引き継ぐ:
  - `point_level_manifest.json`（point-level contract）
  - `train/outputs/feature_contract.json`（学習時固定契約）
  - `model_state.feature_contract_path`（推論時参照）
- `preprocessing` は再現性メタを成果物として固定:
  - `preprocess_bundle.json`（`feature_contract_hash`, `inverse_ready` を含む）
  - `preprocess_report.json`（分布診断 + 再構成誤差）
- Inference のテンプレート選択は `inference.template_run_id` で明示可能。
  - 未指定時は `run_id` 昇順で deterministic に選択。
- `train.mode=sparse_distill` では `sparse_model_profile=small|base|large` を受け、
  checkpoint に `model_architecture`（hidden_channels/num_blocks/dropout/residual）を保存する。
- 精度優先プロファイル（`configs/example_accuracy.toml`）は `strict_split=true` を既定とし、
  valid split が作れない実行は失敗させる（リーク防止）。
- `train` は OOD二層スコアの参照統計を `train_ood_reference.json` に保存し、
  推論時は `model_state.ood_reference` / `train_ood_reference_json` から閾値を再利用する。
- `train` 可視化成果物:
  - `runs/<run_id>/train/outputs/viz/learning_curves.png`
  - `runs/<run_id>/train/outputs/viz/scatter_gt_pred.png`
  - `runs/<run_id>/train/outputs/viz/train_r2.json`
  - `runs/<run_id>/train/outputs/viz/visualization_manifest.json`
- 外付け holdout 可視化成果物:
  - `runs/dataset_3d_test2_pilot/eval/*/images/{xy,xz,yz}/...png`
  - `runs/dataset_3d_test2_pilot/eval/*/visualization_manifest.json`
  - `runs/dataset_3d_test2_pilot/eval/*/report_index.json`
- Optimize plugin の公式例は:
  - `engine=plugin:wafer_surrogate.optimization.plugins.botorch_plugin:run_plugin_search`
  - botorch 未導入時は `builtin:grid` にフォールバックし、`fallback_reason` を保存。

## 10) Accuracy-first再編（実装反映）
- 共通処理の重複を削減するため、`src/wafer_surrogate/core/` を追加。
  - `io_utils.py`: JSON/CSV I/O, run_id sanitize, frame集約
  - `rollout.py`: predict_phi / predict_vn / fallback rollout の共通経路
  - `ood.py`: 条件空間 + 特徴空間の二層OOD評価
    - `condition_score`
    - `feature_score`
    - `combined_status`
    - 各空間で `mahalanobis_distance` / `knn_distance`
- 最適化実行は `src/wafer_surrogate/optimization/engines/` を追加し、engine抽象で切替。
  - `builtin.py`: random/grid/bo/mfbo
  - `optuna_engine.py`: optional optuna
  - `plugin.py`: `plugin:<module[:function]>` 拡張
  - optimize履歴は `best_so_far` / `improvement` / `feature_importance` を保持
- verify は `src/wafer_surrogate/verify/checks/` へ分割し、`python -m wafer_surrogate.verify` は package entrypoint で維持。
- train/inference は external artifact 入力を受け、stage単体実行を可能化。
- inference stage は mode resolver（`single|batch|optimize`）を backend 抽象として分離し、実行モード追加時の変更範囲を局所化する。
- manifest は `schema_version/stage_dependencies/stage_inputs/runtime_env/seed_info/split_info/warnings` を保持する。
- `split_info` は最低限 `num_train_runs/num_valid_runs/leak_checked/reason/loader_mode` を保持する。
- synthetic データは `config[data].dimension=2|3` と `grid_depth` で 3D 生成を切替可能。
