# CONTEXT — ドメイン前提とI/O契約（ウェハ加工形状サロゲート）

この文書は、本チャットで合意した内容を「実装がブレない」粒度で固定する。
未確定項目は **TBD** として明示し、`tasks/tasks.json` の `type=decision` にリンクする。

---

## 1) 目的（Must）
- モンテカルロ（粒子フラックス）× voxel表面の3D形状シミュレーションを代替するサロゲートを構築し、
  - (M1) 計算速度を大幅に向上（シミュレーションレベルの一致）
  - (M2) 実測SEM形状に合わせてブラックボックスパラメータ（表面反応/フラックス）を早期同定
  - (M3) 課題原因特定（表面形状/フラックス/反応など）
  - (M4) レシピ探索（装置入力設定で理想形状）を高速化

## 2) 現状シミュレーションの特徴（前提）
- 反応は MC/kMC 系（エッチ、デポ、スパッタ、反射、脱離による二次反応、表面拡散等を含む）。
- 表面は voxel 表現、フラックスは粒子。
- 3D 1条件は 2時間以上＋メモリ大。
- 形状は基本「特定形状に特化」して計算（時に複数形状）。
- 実測は主に SEM 画像（加工後形状）。丸み・微細形状・マスク境界合わせ込みが難しい。
- 工程ごとの実測条件は おおむね10条件程度。

## 3) サロゲートが目指す状態表現（推奨）
- 形状状態は **SDF（Signed Distance Function）** φ(x,t) を基本とする。
  - φ=0 が表面。
  - 既定の符号: **固体内部が負 (φ<0), 空隙/ガスが正 (φ>0)**。
  - 現行実装では上記符号を canonical とし、符号切替は未提供（変更が必要な場合は decision を追加して拡張する）。
- 時間発展はレベルセット更新を基本とする：
  - φ_{t+Δt} = φ_t - Δt * Vn(x,t) * |∇φ_t|
  - Vn は表面法線速度（総和でも成分分解でも可）。

## 4) 入力と潜在変数（測れないものの扱い）
- 入力条件は2層に分ける。
  1) **制御可能な上位ノブ**（圧力、RF、温度、ガス比など）を低次元ベクトル c として保持。
  2) **測れない/未同定の要素**（フラックス分布詳細、反応確率、足りない物理など）は latent z に集約。
- 実測吸収は「モデル本体を壊さず latent を同定して吸収」を優先。
  - 同定法は MAP（最適化）から開始し、必要なら SBI（posterior）へ。

## 5) 表面モデルで内部影響を含める（学習の工夫）
- 学習時のみ利用できる **特権情報（内部ログ）** を Teacher に入力し、Student へ蒸留する。
  - 例: 直接フラックス/反射フラックス、入射角・エネルギーヒスト、再入射回数統計、機構別寄与。
- 推論時は表面特徴（SDF narrow band、見通し/遮蔽特徴など）と条件(c,z)で動作。

## 6) データスキーマ（シミュレーション）
- データ単位：
  - run = 1つの初期形状 + 1つの条件（c と内部パラメータ）
  - step = 時刻/進行度合いのスナップショット
- 保存形式（推奨）：HDF5 または Zarr
- 既定スキーマ（narrow band で疎に保存する想定）：
  - /runs/{run_id}/meta/recipe : float32 [C]
  - /runs/{run_id}/meta/dt : float32 [1]
  - /runs/{run_id}/steps/{k}/coords : int32 [N,3]
  - /runs/{run_id}/steps/{k}/feat : float16 [N,F]  (φ, ∇φ, κ, material/mask, optional visibility)
  - /runs/{run_id}/steps/{k}/vn_target : float16 [N,1]
  - /runs/{run_id}/steps/{k}/priv : float16 [N,P] (Teacher専用・任意)
- 実装上の必須制約（`src/wafer_surrogate/data/io.py` で検証）：
  - `runs` は1件以上、`run_id` は一意。
  - 各 run の `steps` は1件以上。
  - 各 step で `coords/feat/vn_target` の行数 N は一致する。
  - `coords` は3列固定、`vn_target` は1列固定、`priv` は存在時のみ任意列。
  - `meta/dt` は正の値を1要素で保持する。
- 永続化バックエンド：
  - `h5py` / `zarr` が利用可能なら HDF5/Zarr へ read/write する。
  - 依存未導入時は optional backend として扱い、I/O層は graceful degrade する。

## 7) データスキーマ（SEM実測）
- 入力はSEM画像だが、学習・同定ではまず **輪郭/特徴量** を推奨。
  - 例: CD(複数高さ), sidewall angle, bottom curvature, footing 指標など。
- 推奨形式：
  - raw image: .tif/.png + メタ（倍率、スケール、断面方向）
  - contour: polyline JSON または binary mask PNG
  - derived features: float32 ベクトル y

## 8) 決定済み仕様（ADR連携）
- (DECISION-P1-004 / ADR-0005) 単位系と `Δt`:
  - canonical は `voxel-step`（SDFとVnは voxel 基準）
  - 物理単位（nm, s）への変換は optional metadata (`voxel_pitch_nm`, `dt_seconds`) で保持
  - `Δt` は既定で progress step semantics（必要時のみ physical time へ写像）
- (DECISION-P1-005 / ADR-0006) 材料ラベル:
  - fixed enum + integer ID canonical
  - `material_unknown=0, material_si=1, material_sio2=2, material_sin=3, material_pr=4`
  - unknown は `material_unknown` へ写像し warning を記録
- (DECISION-P1-006 / ADR-0007) SEM観測形態:
  - canonical は cross-section
  - top-view/multi-view は拡張モダリティとして同一契約へ追加可能
- (DECISION-P1-001 / ADR-0002) リポジトリ方針:
  - 既存repo継続（Option B）を Accepted
