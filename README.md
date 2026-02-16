# wafer-surrogate

ウェハ加工形状（SDF/level-set）を対象にした、サロゲート学習・推論・最適化の実験基盤です。
現行実装は、`Data Cleaning -> Featurization -> Preprocessing -> Train -> Inference` の stage 構成と、CLI互換を維持したワークフロー実行を提供します。

## 1. 現在のモデル実装
`src/wafer_surrogate/models/api.py` でレジストリ登録されています。

- `baseline_vn_constant`
  - 定数法線速度 `Vn` を予測する最小ベースライン。
- `baseline_vn_linear`
  - 条件変数の線形結合 + バイアスによる `Vn` 予測（固定重み運用向け）。
- `baseline_mean`
  - 互換エイリアス（内部的には定数モデル）。
- `baseline_vn_linear_trainable`
  - 学習可能な線形 `Vn` モデル（勾配ベース更新）。
- `surface_graph_vn`
  - 表面近傍グラフ情報を利用した `Vn` 予測モデル。
- `operator_time_conditioned`
  - 条件と時刻を用いた演算子近似で `phi(t)` を直接予測可能。
- `sparse_vn_student` / `sparse_vn_teacher`
  - 互換エイリアス（legacy線形実装）。明示名は `*_linear_legacy`。
  - deprecated alias として警告を出力し、`2026-06-30` 以降の削除予定。
  - `train.mode=sparse_distill` は別経路で SparseTensor 学習（依存不足時は fallback）。

補足:
- 正規I/Fは `predict_vn(phi, conditions, step_index)`。
- 互換のため `predict(...)` も維持しています。

## 2. 解析ワークフロー概要

### Data Cleaning
`src/wafer_surrogate/pipeline/stages/cleaning.py`
- 重複除去、欠損補完、レシピ値クランプ、正規化。
- 出力:
  - `cleaned_dataset.json`
  - `cleaning_report.json`

### Featurization
`src/wafer_surrogate/pipeline/stages/featurization.py`
- `target_mode` を切替可能:
  - `frame_mean_delta`（互換）
  - `vn_narrow_band`（局所教師）
- `vn_narrow_band` 時は point-level 特徴を生成:
  - `phi`, `|grad phi|`, `curvature proxy`, `band distance`
  - `coord_x/y/z`, `step_index`
- 出力:
  - `features.csv`, `targets.csv`
  - `reconstruction_bundle.json`
  - `narrow_band_manifest.json`（backend, band設定, 点数）
  - `point_level_manifest.json`（feature contract, recipe keys, feat_dim）
- `frame_mean_delta` 時は SDF統計特徴を生成:
  - `phi_mean/std/min/max`
  - `neg_fraction`, `narrow_band_ratio`
  - `grad_abs_mean/max`
  - `curvature_proxy_mean/max`
  - 時間特徴（`step_fraction`, lag系など）

### Preprocessing
`src/wafer_surrogate/pipeline/stages/preprocessing.py`
- 前処理パイプライン（steps）を適用。
- `feature_transform`:
  - `none|standard|robust|quantile|pca`
- `feature_log1p` と `target_transform`（`identity|standard|log1p`）をサポート。
- 入力特徴と目的変数を別管理で変換し、逆変換メタを `preprocess_bundle.json` に保存。
- `preprocess_bundle.json` には再現性キーを保存:
  - `schema_version`
  - `feature_contract_hash`
  - `inverse_ready`
- `preprocess_report.json` で分布診断と再構成誤差を出力:
  - 列ごとの `mean/variance/missing_rate/outlier_rate`
  - `reconstruction_error.feature_mae/target_mae`
- 出力:
  - `processed_features.csv`
  - `processed_targets.csv`
  - `preprocess_bundle.json`
  - `preprocess_report.json`

### Train
`src/wafer_surrogate/pipeline/stages/train.py`
- 単一/複数モデル変種を学習し比較。
- `run_id` 単位の KFold CV（リーク検知付き）。
- メトリクス（MAE/RMSE）を long-format でも保存。
- `mode=sparse_distill`:
  - optional依存（`torch` + `MinkowskiEngine`）で SparseTensor 蒸留を実行。
  - 未導入時は `fallback_model` へ自動フォールバックし、manifest warningに理由を記録。
  - `sparse_model_profile = small|base|large` で ME ネットワーク深さ/幅を切替。
  - 学習安定化キー:
    - `early_stopping_patience`
    - `grad_clip_norm`（`grad_clip` 互換）
    - `lr_scheduler = none|cosine`
- 学習時の OOD 参照統計を `train_ood_reference.json` に出力（condition/feature の閾値推奨値を含む）。
- leaderboard の優先順位は `student_mae` を第一、`rollout_short_window_error` を第二キーとして保存。
- 出力:
  - `model_state.json`
  - `feature_contract.json`
  - `train_ood_reference.json`
  - `train_metrics.json`
  - `train_model_comparison.csv`
  - `train_metrics_long.csv`
  - `distill_metrics.json`
  - `distill_metrics_long.csv`

### Inference
`src/wafer_surrogate/pipeline/stages/inference.py`
- モード:
  - `single`
  - `batch`
  - `optimize`
- OOD判定: 条件空間 + 特徴空間の二層判定。
- `train_ood_reference.json`（または `model_state.ood_reference`）がある場合、推論時 OOD 判定は学習時統計由来の閾値を優先利用。
- `inference.template_run_id` で rollout テンプレート run を明示指定可能（未指定時は `run_id` 昇順先頭）。
- OOD report 主要項目:
  - `condition_score`, `feature_score`, `combined_status`
  - `condition_threshold`, `feature_threshold`
  - 各空間で `mahalanobis_distance`, `knn_distance`
- `optimize` は `random/grid/bo/mfbo` を標準実装し、`optuna`/`plugin` を optional として扱います。
- `optimize` 履歴には `best_so_far`, `improvement`, `feature_importance` を保存します。
- `inference.calibration` は observation plugin (`observation_model`) を指定可能です。
- `latent_prior_json` は `z_init/prior_mean/prior_std` の数値配列 schema を検証します。

## 3. 形状評価（Observation/Metrics）

### 観測特徴
`src/wafer_surrogate/observation/registry.py`, `src/wafer_surrogate/observation/plugins/baseline.py`
- 2D/3D shape から観測特徴を抽出。
- 統計特徴に加えて、形状評価向け特徴を実装:
  - `cd_top_ratio`, `cd_mid_ratio`, `cd_bottom_ratio`
  - `sidewall_angle_deg`
  - `centerline_curvature_proxy`
  - `footing_proxy`
  - `grad_abs_mean_2d`, `grad_abs_max_2d`, `narrow_band_ratio_2d`

### 評価指標
`src/wafer_surrogate/metrics.py`
- ロールアウト評価:
  - `sdf_l1_mean`, `sdf_l2_rmse`
  - `vn_mae`, `vn_rmse`
  - プロファイル誤差
  - 観測特徴誤差（feature単位 MAE/RMSE を含む）

### Prior
`src/wafer_surrogate/prior/registry.py`, `src/wafer_surrogate/prior/plugins/gaussian.py`
- prior は plugin registry 経由で解決されます。
- `score_latent` は finite float を返す契約で検証されます。

## 4. 出力構成

パイプライン実行時:
- `runs/<run_id>/<stage>/configuration/`
- `runs/<run_id>/<stage>/logs/`
- `runs/<run_id>/<stage>/outputs/`
- `runs/<run_id>/manifest.json`
- `manifest.json` は `stage_dependencies/stage_inputs/runtime_env/seed_info/split_info/warnings` を保持
- `runs/<run_id>/leaderboard/`
  - `data_path/leaderboard.csv`
  - `model_path/leaderboard.csv`
  - `viz/`（可視化。matplotlib未導入時はCSV fallback）
- `runs/<run_id>/train/outputs/viz/`
  - `learning_curves.png`
  - `scatter_gt_pred.png`
  - `train_r2.json`
  - `visualization_manifest.json`
- holdout評価（外付けscript）
  - `runs/dataset_3d_test2_pilot/eval/*/visualization_manifest.json`
  - `runs/dataset_3d_test2_pilot/eval/*/report_index.json`

## 5. 実行コマンド（SSOT）

`python` 直実行ではなく `scripts/commands.sh` を使用します。

- CLIヘルプ
```bash
./scripts/commands.sh run -- --help
```

- クイック検証
```bash
./scripts/commands.sh verify -- --quick
```

- フル検証
```bash
./scripts/commands.sh verify -- --full
# or
./scripts/commands.sh verify-full
```

- 生成物クリーンアップ
```bash
./scripts/cleanup_generated.sh
./scripts/cleanup_generated.sh --remove-tmp-venv
```

- pipeline実行
```bash
./scripts/commands.sh run -- pipeline run --config configs/example.toml
./scripts/commands.sh run -- pipeline run --config configs/example.toml --stages cleaning,featurization,preprocessing,train,inference
./scripts/commands.sh run -- pipeline run --config configs/example_accuracy.toml
```

- `ataset_3d_test2` を YAML 1枚で実行（prepare -> train -> eval -> compare）
```bash
python3 scripts/run_dataset_3d_test2_from_yaml.py \
  --config configs/dataset_3d_test2_train_eval.yaml
```

- holdout評価（SDF断面 + report index）
```bash
python3 scripts/eval_dataset_3d_test2_holdout.py \
  --model-state runs/dataset_3d_test2_pilot_quick/train/outputs/model_state.json \
  --holdout-json runs/dataset_3d_test2_pilot/prepared/holdout_dataset.json \
  --out-dir runs/dataset_3d_test2_pilot/eval/quick_primary \
  --viz-config-yaml configs/visualization.default.yaml
```

- 3DメッシュHTML（YAML制御）
```bash
python3 scripts/visualize_hole_mesh_3d.py \
  --model-state runs/dataset_3d_test2_pilot_quick/train/outputs/model_state.json \
  --holdout-json runs/dataset_3d_test2_pilot/prepared/holdout_dataset.json \
  --split-manifest runs/dataset_3d_test2_pilot/prepared/split_manifest.json \
  --out-dir runs/dataset_3d_test2_pilot/eval/quick_primary/mesh3d \
  --viz-config-yaml configs/visualization.default.yaml
```

- 最適化探索
```bash
./scripts/commands.sh run -- search-recipe --config configs/example.toml --trials 30 --engine grid
./scripts/commands.sh run -- search-recipe --config configs/example.toml --trials 30 --engine bo
./scripts/commands.sh run -- search-recipe --config configs/example.toml --trials 30 --engine mfbo
./scripts/commands.sh run -- search-recipe --config configs/example_accuracy.toml --engine plugin:wafer_surrogate.optimization.plugins.botorch_plugin:run_plugin_search --trials 20
```

## 6. 設定ファイル
`src/wafer_surrogate/config/loader.py`
- `.toml` を標準サポート。
- `.yaml/.yml` は `pyyaml` がある場合に読み込み可能。
- 推奨プロファイル:
  - `configs/example.toml`: 互換優先（`frame_mean_delta` + `train.mode=tabular`）
  - `configs/example_accuracy.toml`: 精度優先（`vn_narrow_band` + `train.mode=sparse_distill` + calibration有効 + `preprocessing.feature_transform=robust` + `strict_split=true`）
    - `strict_split=true` のため、学習runが1件しかない設定では意図的に fail します（リーク防止）。
- 可視化プロファイル:
  - `configs/visualization.default.yaml`（既定: 全可視化ON）
  - 優先順位: `CLI --viz-config-yaml` > `stage.visualization` > `run.visualization` > `configs/visualization.default.yaml`
  - 依存不足（matplotlib/plotly）は fail せず `visualization_manifest.json` に skip 理由を保存。
- synthetic 3D データ設定（`config[data]`）:
  - `dimension = 2|3`
  - `grid_depth = 16`（`dimension=3` のとき有効）

### 6.1 `ataset_3d_test2` 用 YAML
- 実行全体設定: `configs/dataset_3d_test2_train_eval.yaml`
- 学習設定: `configs/dataset_3d_test2_pilot_train.yaml`
- 可視化設定: `configs/visualization.default.yaml`

主要キー（`configs/dataset_3d_test2_train_eval.yaml`）:
- `paths.data_dir`: 入力データディレクトリ（既定 `ataset_3d_test2`）
- `paths.out_root`: 出力ルート（prepare/eval比較/レポート保存先）
- `paths.train_config`: 学習設定ファイル（trainハイパラはここで編集）
- `paths.viz_config`: 可視化トグルYAML
- `split.train_runs`: 学習に使う run 一覧
- `split.primary_holdout_run`: 主評価 holdout（ゲート判定）
- `split.secondary_holdout_run`: 追認 holdout（主評価通過後）
- `prepare.band_width`: narrow-band 幅
- `prepare.min_grad_norm`: Vn教師計算の最小勾配
- `prepare.target_material_id`: ターゲット材ID（`null`で自動選択）
- `prepare.target_selection_mode`: `auto_hole|auto_dynamic`
- `prepare.domain_boundary_margin_vox`: 境界近傍の学習点除外幅
- `prepare.phi_boundary_clip_vox`: SDF計算前のValidMask侵食幅
- `prepare.include_terminal_step_target`: 最終step教師を学習に含めるか
- `evaluation.analysis_xy_margin_vox`: 評価時XYマージン
- `evaluation.analysis_z_margin_vox`: 評価時Zマージン
- `runtime.run_real_me`: real ME経路を実行するか（`0|1`）
- `postprocess.mesh3d.*`: 3Dメッシュ可視化のON/OFFと表示設定

主要キー（`configs/dataset_3d_test2_pilot_train.yaml`）:
- `train.mode`: `sparse_distill|tabular`
- `train.strict_split`: runリーク防止（`true`推奨）
- `train.teacher_epochs`, `train.student_epochs`
- `train.learning_rate`, `train.weight_decay`
- `train.distill_alpha|beta|gamma`
- `train.rollout_loss_enabled`, `train.rollout_k`, `train.rollout_weight`
- `train.temporal_step_weight`: `uniform|early_decay`
- `train.step_sampling_policy`: `uniform|early_bias`
- `train.sparse_model_profile`: `small|base|large`

実行結果の確認:
- 全体レポート: `runs/dataset_3d_test2_pilot/eval/report_index.json`
- holdout評価: `runs/dataset_3d_test2_pilot/eval/quick_primary/holdout_eval.json`
- 可視化manifest: `runs/dataset_3d_test2_pilot/eval/quick_primary/visualization_manifest.json`
- 3Dメッシュ: `runs/dataset_3d_test2_pilot/eval/quick_primary/mesh3d/`

### 6.2 仮想環境構築（推奨）
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# 実行に必要な主要ライブラリ（未導入の場合）
python -m pip install numpy scipy matplotlib pyyaml plotly h5py
```

## 7. 関連ドキュメント
- 実装ハンドオフ: `README_HANDOFF.md`
- 可視化補足: `README_VIZ_ADDON.md`
- 設計詳細: `docs/ARCHITECTURE.md`

## 8. verify --full について
- `verify --quick` は依存なし基準で、`MinkowskiEngine` はローカル互換 shim（`src/MinkowskiEngine/`）でも通過可能です。
- `verify --full` は optional 依存の実装経路を必須化し、**MinkowskiEngine は実パッケージ必須（shim不可）**です。
- `verify --full` 実行時は内部的に `WAFER_SURROGATE_REQUIRE_REAL_ME=1` を有効化し、shim が検出された場合は失敗します。
- 実機で CUDA/本家 MinkowskiEngine を使う場合は、shim より優先される実パッケージを環境へ導入してください。
- macOS での推奨構成は `torch==2.2.2` 固定です（ME互換運用）。
- 依存状況は以下で確認できます。

```bash
./scripts/commands.sh verify -- --quick
```

出力の `runtime capabilities` に不足依存が表示されます。

### 8.1 macOS で optional-full 依存を揃える
`openblas` と `libomp` を Homebrew で導入済みの前提で、以下を実行します。

```bash
bash scripts/install_optional_full_macos.sh
./scripts/commands.sh verify -- --full
```

### 8.2 torchmd-net-cpu の競合整理（削除/分離）
`torchmd-net-cpu` は別ワークロードで `torch==2.7.1` を要求する場合があり、  
本プロジェクトの ME 互換固定（`torch==2.2.2`）と競合します。

状態確認:
```bash
./scripts/commands.sh torchmd status
```

分離運用（推奨）:
```bash
./scripts/commands.sh torchmd isolate --venv .venv_torchmd --skip-install
```
`--skip-install` は「環境だけ分離」するモードです。外部 `torchmd-net-cpu` ソースのビルド可否に依存せず実行できます。

分離後に現在環境から削除:
```bash
./scripts/commands.sh torchmd isolate --venv .venv_torchmd --skip-install --remove-from-current
# または削除のみ
./scripts/commands.sh torchmd remove-current
```

選択基準:
- このリポジトリだけ使う: `remove-current`（削除）を推奨。
- torchmd も継続利用する: `isolate`（別venv分離）を推奨。
