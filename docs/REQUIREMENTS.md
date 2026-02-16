# REQUIREMENTS — 要件一覧（漏れ禁止）

この要件は本チャットで検討した内容を網羅し、実装タスクへ **100%トレース**されます。

- **Must**: 実装が必ず満たすべき
- **Should**: 可能な限り満たす（P1/P2）
- **Could**: 余力があれば

---

## Must（必須）

### R-MUST-001 形状時系列サロゲート
入力（初期形状＋条件）から、加工形状の**時間発展**を出力できること。

### R-MUST-002 形状連続性担保
形状表現・更新により、表面の連続性（破綻しない形状）を担保できること。

### R-MUST-003 少データ前提の設計
問題ごとに膨大な学習データを必要としない設計（局所サンプル化・能動学習・転移などの導入余地）を持つこと。

### R-MUST-004 入力変数爆発への対策
装置ノブ/シミュレータ入力が多い前提で、入力を **(a) 制御ノブ c** と **(b) 潜在 z** に整理できること。

### R-MUST-005 潜在変数による実測吸収
フラックス計測困難・表面反応データ不足を前提に、未知要因を latent に集約し、実測SEMに合わせて同定・吸収できること（MAP→必要ならSBIへ拡張可能）。

### R-MUST-006 目的に応じたモジュール分割
「特長量化 / 前処理 / 学習 / 推論 / 評価」を**独立に実行・再利用**できる設計にすること。

### R-MUST-007 パイプライン合成と比較
特長量化・前処理・学習を、用途に応じて**パイプラインとして組み替え**、複数組み合わせを比較評価できること（順序固定はしない）。

### R-MUST-008 複数メトリクス評価
学習・推論の評価は単一指標にせず、複数メトリクスを実装し、比較できること。

### R-MUST-009 推論のin-domain判定（OODフック）
推論入力が学習範囲内かどうかを評価できるよう、OOD判定のフックを持つこと。  
少なくとも `condition_score` と `feature_score` の二層スコアを出力し、`combined_status` で統合判定できること。

### R-MUST-010 推論は単一・複数条件に対応
単一条件だけでなく、複数条件（バッチ）で推論し、結果サマリーを出力できること。

### R-MUST-011 レシピ探索（最小実装）
サロゲートを使ってレシピ探索を行えること（P0はランダム探索等の最小実装で可、BO/MFBOは拡張）。

### R-MUST-012 生成物管理
実行生成物（ログ、モデル、図、状態）は `runs/` に集約し、git管理しない（.gitignore）。

### R-MUST-013 実行/検証コマンドの単一の正（SSOT）
run/test/verify コマンドを `scripts/commands.sh` に集約し、文書・タスクから参照すること。

### R-MUST-014 preflight による環境差吸収
python vs python3、src import、MPLCONFIGDIR 等の事故を `scripts/preflight.sh` と `scripts/run.sh` で抑制すること。

### R-MUST-015 Codex autorun のresume
自動実行は state（runs/autorun_state.json）で完了タスクを記録し、再実行で続きから再開できること（最初からやり直し禁止）。

### R-MUST-016 codex exec フラグ検出
`codex exec --help` を解析し、read-only事故を避けるため workspace-write 相当のフラグを自動選択できること。

### R-MUST-017 checkpoint 方針
checkpoint は原則 P0 末尾のみ（必要が生じた場合は decision で変更）。

### R-MUST-018 未決は decision タスク化
仕様・方針が未確定の場合、勝手に追加せず `type=decision` タスクとして停止できること。

### R-MUST-019 基本処理単位の標準化
基本処理単位を `Data Cleaning / Featurization / Preprocessing / Train / Inference` として定義し、各処理が独立実行可能であること。

### R-MUST-020 Pipeline 任意組み合わせ実行
任意の処理組み合わせ（例: `preprocessing + train`）で実行できる pipeline 実行機能を持つこと。

### R-MUST-021 Stage成果物の標準ディレクトリ
各 stage は `runs/<run_id>/<stage>/{configuration,logs,outputs}` を標準出力構成として利用すること。

### R-MUST-022 推論モード三分割
Inference は `single / batch / optimize` の3モードを持ち、各モードで結果サマリと可視化情報を出力できること。

### R-MUST-023 leaderboard 標準出力
比較評価結果を `runs/<run_id>/leaderboard/{data_path,model_path}` に出力できること。

### R-MUST-024 run manifest
`runs/<run_id>/manifest.json` に stage依存、実行状態、成果物参照を記録できること。

### R-MUST-025 特徴量契約（feature contract）の固定
`Featurization -> Train -> Inference` で、特徴次元・順序・条件キーの契約を固定し、学習/推論で不整合が起きないこと。  
少なくとも `point_level_manifest.json` と `feature_contract.json` を成果物として保持し、`model_state` から参照できること。
契約不一致（列名/順序/dim）は hard fail し、警告で握りつぶさないこと。
前処理成果物は `feature_contract_hash` を保持し、再現性検証に利用できること。
精度優先設定（`configs/example_accuracy.toml`）は `strict_split=true` を既定とし、valid split 不可時は hard fail すること。

### R-MUST-026 OODレポートの標準スキーマ
OODレポートに `condition_score`, `feature_score`, `combined_status` を保持し、各空間で距離内訳（例: Mahalanobis近似, kNN）を記録できること。
加えて、学習時に保存した参照統計（`train_ood_reference.json`）を推論時に参照し、
`condition_threshold` / `feature_threshold` を一貫ロジックで適用できること。

### R-MUST-027 synthetic 3D 生成の設定駆動対応
合成データ生成は 2D/3D を設定で切替可能であること。  
少なくとも `config[data].dimension` と `config[data].grid_depth` を受け、pipeline本線で利用できること。

### R-MUST-028 optimize 履歴の収束可視化情報
`optimize` の履歴に、収束確認のための `best_so_far` と `improvement` を記録できること。  
可能な場合は特徴寄与（feature importance）も併せて出力できること。

### R-MUST-029 指標命名の正式化
成果物の評価指標に placeholder 命名を残さず、実測値として解釈可能な正式名称（例: `vn_mae`, `vn_rmse`）で出力すること。

### R-MUST-030 verify quick/full 契約
`verify --quick` は optional依存なし環境を基準に pass できること。  
`verify --full` は optional依存（MinkowskiEngine/sbi/botorch/optuna等）経路を必須化し、不足時は fail すること。  
MinkowskiEngine は `--quick` では shim 許可、`--full` では実パッケージ必須（shim は fail）とすること。
加えて、`__pycache__/`, `*.pyc`, `.venv_torchmd_tmp/` の tracked 生成物がないことを verify で検証可能であること。

### R-MUST-031 observation/prior plugin 拡張点
観測モデルと latent prior は registry + plugin 構成で実装し、baseline 実装から差し替え可能であること。  
`score_latent` は finite float 返却契約を満たし、違反時に明示エラーを返すこと。

---

## Should（推奨）

### R-SHOULD-001 SDF（narrow band）データスキーマ
SDF + narrow band の疎表現（coords + feat + vn_target）を主要データスキーマとして扱えること。

### R-SHOULD-002 Vn教師の自動生成
SDF差分と勾配から Vn の擬似教師を生成し、局所学習でデータ効率を上げられること。

### R-SHOULD-003 level-set 更新
Vn から φ を更新する level-set ループを持ち、ロールアウト学習（短窓unroll）に拡張できること。

### R-SHOULD-004 内部影響の学習（特権情報→蒸留）
学習時のみ内部ログ（フラックス統計、反射寄与等）を使い、推論時は不要にする teacher-student 蒸留を実装できること。

### R-SHOULD-005 表面モデル（非局所結合）
表面点群/メッシュ上のモデル（GNN/Transformer）で、遮蔽・反射など非局所影響を表現できる拡張経路を持つこと。

### R-SHOULD-006 多忠実度（LF/HF）導入余地
低忠実度（粗格子/簡略物理/2D断面）と高忠実度（3D MC）を併用し、学習・最適化を効率化できる設計余地を持つこと。

### R-SHOULD-007 事前学習→微調整（PEFT余地）
工程ごとの運用コストを下げるため、事前学習/微調整（小さな差分更新）の導入余地を持つこと。

### R-SHOULD-008 SEM前処理（輪郭抽出→特徴）
SEM画像から輪郭/特徴量を安定に抽出し、学習・同定に使えるI/O契約を持つこと。

### R-SHOULD-009 出力可視化
用途に応じた出力図（断面プロファイル、CD/角度、誤差マップ等）を複数作れること。

---

## Could（任意）

### R-COULD-001 SBI（posterior）実装
MAP同定をSBI（posterior分布推定）へ拡張できること。

### R-COULD-002 BO/MFBO（BoTorch等）
レシピ探索を BO/MFBO へ拡張できること。

### R-COULD-003 Neural Operator 系
φ0,c,t → φ(t) を直接学習する Neural Operator をベースラインとして追加できること。

### R-COULD-004 微分可能レンダリング
SDF→画像の観測モデルでSEM画像空間の損失を導入できること。

### R-COULD-005 生成モデル（拡散）prior
形状の多解性を扱うための生成priorを導入できること。

## VIZ add-on

（注）詳細は `docs/VIZ_REQUIREMENTS_ADDON.md` を参照。
