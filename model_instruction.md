以下は、これまでの前提（**MC粒子フラックス×voxel表面、3Dが重い、入力変数が多い、SEM形状しか実測がない、工程ごと実測は10条件程度、時間発展が必要、形状連続性が必要、学習データ爆増は避けたい**）を踏まえたうえで、**有用度が高い順に 1 / 2 / 3** を選び、**そのままコード実装に落とせる設計（データ仕様・モデル・学習・推論）**を具体化します。
（※「手法単体」ではなく、**実装可能な“組み合わせ”**として提示します。）

---

# まず結論：有力順 1 / 2 / 3

## 1位（最推奨）

**SDF（narrow band）＋学習した法線速度 (V_n)（閉包モデル）＋レベルセット更新**
＋（学習時のみ）**特権情報（内部ログ）Teacher→Student蒸留**
＋（実測吸収）**latent（未知フラックス/反応）同定（SBI or MAP）**

* **狙い**：高速・連続性・少データ・原因分解・実測吸収が全部両立しやすい
* **疎テンソル実装**が実用的（MinkowskiEngineが代表的に疎テンソルを基本表現としている） ([nvidia.github.io][1])
* **レベルセットでの高速形状進化**は、既存のトポグラフィ進化フレームワーク（ViennaPS）でも高性能設計が確認できる ([サイエンスダイレクト][2])
* 実測吸収の「尤度が書けない」問題は **SBI** が現実解（sbi reloaded が整備） ([joss.theoj.org][3])

---

## 2位（表面データ中心でメモリを極小化しつつ内部影響を入れる）

**表面点群/メッシュのGraph Transformer（非局所結合つき）で (V_n) を予測**
＋ **SDFへ投影してレベルセット更新**（位相変化も安全）
＋ Teacher→Student蒸留（内部ログ→見通し/遮蔽などの表面特徴へ）

* **狙い**：表面だけで学習したい・内部影響（遮蔽/反射/再入射）を“非局所”として入れたい
* ただし実装難度は1位より上がりやすい（表面⇄SDF変換、非局所エッジ設計）

---

## 3位（汎用化・多形状展開を狙う研究開発向き）

**Time-conditioned Neural Operator（場→場）で (\phi(t)) を直接予測（rollout不要）**
＋ **多忠実度（LF大量＋HF少量）＋PINO/制約**
＋ 実測はPEFT（小さなアダプタ）＋ latent同定

* **狙い**：形状や条件の横展開（工程間転移）を強く狙う
* ただし **データ設計と安定化が難しく、最短で成果を出す目的には1位が勝ちやすい**
* 3位は「長期的に“汎用サロゲート基盤”へ育てる」位置づけ

---

# 共通：あなたのサロゲート実装で最初に固定すべき“仕様”

この3案すべてで共通に効くので、最初に決め打ち推奨です。

## A. 形状状態は「SDF（符号付き距離）(\phi)」を採用

* 表面：(\phi=0)
* 内外：符号で区別
* 形状連続性、法線、曲率が取りやすい
* レベルセット更新で位相変化も自然

## B. 時間の刻みは「物理時間」より「形状進化量」で刻むのが強い

例：

* 1ステップ＝「面平均で Δh=0.5nm進む」等
  → 学習が安定し、工程間比較もしやすい

## C. 入力を2層に分ける

* **制御できる上位ノブ**：圧力、RF、温度、流量比…（10次元以下に圧縮）
* **未知・測れないもの**：フラックス分布や反応確率の不足物理 → **latent z**（8〜32次元）へ集約
  → 実測SEMで z を同定して吸収する

---

# 1位：SDF narrow band + Sparse CNNで (V_n)（閉包） + レベルセット更新 + 蒸留 + 同定

「最短で動く」「データ爆増しない」「実測吸収できる」を同時に狙う設計です。

---

## 1-1. モデルが解く問題の定義

### Forward（サロゲート）

入力：

* 初期形状 (\phi_0)（SDF）
* レシピ (c)（低次元）
* latent (z)（未知フラックス/反応を代表）
  出力：
* 形状時系列 (\phi_{1..T})（または最終形状）

更新は：
[
\phi_{t+\Delta t}=\phi_t-\Delta t,V_n(x,t),|\nabla \phi_t|
]
ここで **(V_n) をNNが予測**します。

---

## 1-2. 学習データ：シミュレーションから何を保存するか（追加ログ含む）

### 必須（最低限）

* 各runごと

  * recipeベクトル (c)（float32[C]）
  * 時刻/ステップ情報（dt, step index）
* 各ステップ k ごと

  * SDF (\phi_k)（できればSDFそのもの、難しければvoxelから計算）
  * 次ステップ (\phi_{k+1})（教師を作るため）

### 推奨（学習効率が上がる）

* 材料ID / マスクID（離散チャネル）
* 可能なら「機構別寄与」の集計（etch / depo / sputter / reflect…）

  * “粒子ログ全保存”でなくてOK
  * 例：表面パッチごとにイベント回数・堆積量・除去量の統計

### 重要：特権情報（Teacher用、推論では不要）

* 表面近傍パッチごとに

  * 直接フラックス、反射由来フラックスの推定
  * 入射角分布（ヒスト）、エネルギー分布（ヒスト）
  * 反射回数、再入射回数の統計
  * 表面拡散の有効長/滞留時間の統計（可能なら）

これをTeacherだけに見せて、Studentへ蒸留します（運用コストを増やさずに内部効果を入れる）。

---

## 1-3. 教師ラベルの作り方（コード化の要点）

### ターゲット：法線速度 (V_n)

SDF差分から擬似教師を作れます：

[
V_n(x,t)\approx\frac{\phi_t(x)-\phi_{t+\Delta t}(x)}{\Delta t,|\nabla\phi_t(x)|}
]

* これを **narrow band（|φ|<w）だけ**で計算して学習します。
* (|\nabla\phi|) が小さい点（数値不安定）にはマスクを掛ける。

> この作り方にすると「1回の3Dシミュレーション」から大量の局所教師が取れるので、run数を増やさなくても学習が回りやすいです。

---

## 1-4. データ形式（おすすめ）

### 形式：HDF5（またはZarr）

* 大量の「run × timestep × points」を扱うなら

  * **HDF5 + chunk + 圧縮**（gzip / blosc系）
  * 分散ならZarr

### HDF5の具体スキーマ案（narrow band点で保存）

```
/runs/{run_id}/meta/recipe            float32 [C]
/runs/{run_id}/meta/geom_id           int32   [1]
/runs/{run_id}/meta/dt                float32 [1]
/runs/{run_id}/steps/{k}/coords       int32   [N,3]   # narrow band voxel index
/runs/{run_id}/steps/{k}/feat         float16 [N,F]   # phi, grad, kappa, mat_id... (float16推奨)
/runs/{run_id}/steps/{k}/vn_target    float16 [N,1]
/runs/{run_id}/steps/{k}/priv         float16 [N,P]   # Teacherのみ（任意）
```

* `coords` は (x,y,z) の整数格子
* `feat` は float16 でも大抵問題になりにくい（速度と容量に効く）
* **train/val/test splitは run単位**（点単位で混ぜるとリークします）

---

## 1-5. ネットワーク構成（Student/Teacher）

### Student（本番用）

* 入力：SparseTensor（coords + feat）
* 条件：recipe (c) と latent (z) を埋め込み、FiLMまたはconcatで注入
* 出力：(V_n)（または機構別 (V_\text{etch},V_\text{depo},...)）

疎テンソルの基本は MinkowskiEngine の SparseTensor がそのまま使えます ([nvidia.github.io][1])

### Teacher（学習専用）

* Studentと同形状でOK
* 追加で `priv` 特徴を入力に足す（内部ログ）

### 蒸留（Teacher→Student）

* 出力蒸留：Studentの (V_n) をTeacherに寄せる
* 表現蒸留：中間層の埋め込み（embedding）をTeacherに寄せる（より効く）

---

## 1-6. 学習手順（ステージ分けが重要）

### Stage 0：前処理・特徴生成を固める

* voxel→SDF（距離変換）
* grad / curvature の計算
* narrow band抽出

### Stage 1：Teacherを教師ありで学習（privあり）

* ロス：MSE((V_n)) + スムーズ正則化（表面上のラプラシアン等）

### Stage 2：Studentを学習（privなし＋蒸留）

* ロス：

  * MSE((V_n))（通常教師）
  * distill loss（Teacher出力/中間表現）
  * optional：multi-step rollout loss（Kステップ進めたφの誤差）

### Stage 3：rollout安定化（任意だが強い）

* 1ステップだけ合っても、ロールアウトで崩れます
  → `K=3~10` ステップの短いunrollを損失に入れる
  （全Tではなく短窓で良い）

---

## 1-7. 推論（サロゲートシミュレータ）の実装設計

### 推論ループ（概略）

1. 初期voxel→SDF (\phi_0)
2. for step=0..T-1

   * narrow band抽出
   * SparseTensorを作る
   * Studentで (V_n) 推定
   * レベルセット更新で (\phi) 更新
   * 必要ならSDF再初期化（毎回でなく、数ステップに1回でも可）

> レベルセット/幾何更新は ViennaPS のように高性能設計が可能な領域です ([サイエンスダイレクト][2])

---

## 1-8. “実測SEM吸収”の具体設計（latent同定）

実測が10条件でも成立させるために、**モデル本体は固定し、latentだけ動かす**設計にします。

### 観測 y の定義（SEM→特徴）

* 断面輪郭から

  * CD（複数高さ）
  * sidewall angle
  * bottom形状特徴（曲率、フッティング指標）
  * オーバーハング指標（ある場合）
* y は float32[K_feat] にまとめる

### 同定手段（2択）

**(a) MAP（最適化）**

* z を最適化： (\min_z |y(\phi(z)) - y_\text{SEM}|)
* 実装が簡単、ただし多峰性に弱い

**(b) SBI（推奨）**

* sbi は「シミュレータに対する事後分布推定」を行えるパッケージ ([GitHub][4])
* 学習は「(z,c,geom) → y」を大量に生成して posterior (p(z|y,c,geom)) を学習
* sbi reloaded（JOSS）でワークフローが整備されています ([joss.theoj.org][3])

---

## 1-9. 必要データ量の目安（run数）

ここが一番気になる点だと思うので、かなり具体に出します。
※「1 run = 形状1種×条件1」の3D HF MCシミュレーション想定。

### HF（高忠実度）3Dシミュレーション run数

* **最小MVP**：30〜50 run

  * 条件空間を10次元以下に圧縮できている前提
* **標準（実務で安定）**：80〜150 run
* **多形状・多工程へ広げる前提**：各形状あたり 50〜100 run で事前学習→PEFT

### 1 runあたりの時間ステップ

* 推奨：**20〜50 snapshot**

  * 多すぎると保存と前処理が重い
  * 少なすぎると時間ダイナミクスが学べない

### 1 snapshotあたりのnarrow band点数

* 原データは数万〜数十万点になりがち
* 学習は **1 snapshotあたり 1万〜5万点をサブサンプル**で十分回りやすい（patch学習推奨）

### 保存容量目安（例）

* 1 run：20 snapshot × 2万点 × (feat 16ch + target)

  * 数十MB〜数百MB程度（float16中心なら抑えやすい）

---

## 1-10. コード設計（Pythonのモジュール構成案）

```
project/
  configs/
    model_sdf_sparse.yaml
    train.yaml
  src/
    data/
      build_dataset_from_sim.py
      h5_dataset.py
      sem_features.py
    geometry/
      sdf_utils.py
      levelset_update.py
      marching_cubes.py
    models/
      sparse_unet_film.py
      teacher_student.py
    training/
      train_step.py
      losses.py
      rollout_loss.py
    inference/
      simulate_surrogate.py
      calibrate_latent_map.py
      calibrate_latent_sbi.py
    optimization/
      mfbo_botorch.py
  train.py
  eval.py
```

### コード骨子（擬似コード）

```python
# src/data/h5_dataset.py
class NarrowBandDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split, patch_size=64, n_points=20000, use_priv=False):
        ...

    def __getitem__(self, idx):
        # choose run_id, step k, and spatial patch bbox
        coords, feat, vn = load_from_h5(...)
        # optional privileged features
        if self.use_priv:
            priv = load_priv(...)
            feat = np.concatenate([feat, priv], axis=1)
        return coords, feat, vn, recipe_vec

# src/models/sparse_unet_film.py
class SparseUNetFiLM(nn.Module):
    def __init__(self, in_ch, cond_ch, out_ch=1):
        ...
    def forward(self, sparse_tensor, cond_vec):
        # cond_vec -> FiLM params (gamma,beta) per block
        # sparse conv blocks
        return vn_pred_sparse_tensor

# src/geometry/levelset_update.py
def levelset_step(phi, vn_on_grid, dt):
    grad = finite_diff_grad(phi)
    phi_next = phi - dt * vn_on_grid * np.linalg.norm(grad, axis=0)
    return phi_next

# src/inference/simulate_surrogate.py
def simulate(phi0, recipe, z, model, T, dt):
    phi = phi0.copy()
    for t in range(T):
        coords, feat = build_narrow_band_features(phi, recipe, z)
        vn = model(coords, feat, recipe, z)
        vn_grid = splat_to_grid(coords, vn, phi.shape)
        phi = levelset_step(phi, vn_grid, dt)
        if t % REINIT_EVERY == 0:
            phi = reinit_sdf(phi)
    return phi
```

---

# 2位：表面Graph Transformer + 非局所結合 + SDF更新（表面中心・内部影響を入れる）

「表面だけで扱いたい」「内部影響（遮蔽/反射/再入射）を非局所で入れたい」を、実装可能な形に落とした案です。

---

## 2-1. データ仕様（表面点群ベース）

各ステップで “表面点群サンプル” を作る：

* `pos`: float32 [N,3]
* `normal`: float32 [N,3]
* `kappa`: float16 [N,1]
* `mat_id/mask_id`: int16 [N,1]（one-hotでも可）
* `vis_feat`: float16 [N,V]（遮蔽率、開口角、平均自由行程など）
* `vn_target`: float16 [N,1]
* `priv`（Teacher用）：内部ログ由来特徴

### 格納例（HDF5）

```
/runs/{run}/steps/{k}/pos        float32 [N,3]
/runs/{run}/steps/{k}/normal     float16 [N,3]
/runs/{run}/steps/{k}/feat       float16 [N,F]
/runs/{run}/steps/{k}/vn_target  float16 [N,1]
/runs/{run}/steps/{k}/priv       float16 [N,P] (optional)
```

---

## 2-2. グラフ構築（実装がブレやすいので固定案）

### 近傍エッジ（局所）

* kNN（k=16〜32）で `edge_index` を作る
* `edge_attr`: 相対位置、距離、法線内積、材料一致など

### 非局所エッジ（内部影響の近似）

ここがキモで、実装難度を上げずに効かせるコツは次のどちらかです。

**案A：Ray castingで「見通しがある点」だけ非局所接続**

* 各点から少数方向にレイを飛ばし、最初に当たる面点を候補として非局所エッジにする
* エッジ重み＝距離・角度・遮蔽スコア

**案B：Global token（Transformer）で非局所を近似**

* 表面点群をチャンク化し、チャンク間attentionで非局所結合
* 実装は楽だが、解釈は弱め

---

## 2-3. モデル（Torch Geometric想定）

* Node Encoder（MLP）
* Graph Transformer（局所＋非局所）
* Head：(V_n)（または成分分解）

Teacher/Student蒸留は1位と同様（Teacherがprivを見る）。

---

## 2-4. 形状更新（ここを簡単にする）

表面点群を直接動かす（点を (x' = x + dt,V_n n)）だけだと、

* 自己交差
* トポロジ変化
* リメッシュ問題
  が出やすいです。

そこで **推奨**：

* 表面点群で (V_n) を推定
* **SDFグリッドのnarrow bandへ投影してレベルセット更新**
  → 位相変化に強く、1位の幾何更新コードを再利用できる

---

## 2-5. 必要データ量の目安

* **MVP**：HF 20〜40 run
* **標準**：HF 60〜120 run
  （表面だけの方が1 runあたりの学習サンプルは多いが、非局所を学ぶには条件多様性が必要になりやすい）

---

# 3位：Time-conditioned Neural Operator（(\phi_0,c,t \to \phi(t))）+ 多忠実度

「rollout誤差蓄積を避ける」「多形状へ展開」を重視した案です。

---

## 3-1. データ仕様（dense SDFが基本）

* LF：粗い解像度（例 64³）で大量 run
* HF：高解像度（例 128³〜）で少量 run

格納例（Zarr推奨、HDF5でも可）：

```
/runs/{run}/phi0          float16 [X,Y,Z]
/runs/{run}/phi_seq       float16 [T,X,Y,Z]
/runs/{run}/recipe        float32 [C]
/runs/{run}/time_grid     float32 [T]
```

---

## 3-2. 学習タスクの定義（time-conditioning）

学習サンプルは

* 入力：((\phi_0), recipe (c), time (t))
* 出力：(\phi(t))

こうすると

* rolloutが不要（誤差蓄積が減る）
* 任意時刻にクエリできる

---

## 3-3. モデル実装の現実解

* いきなり高度なNOより、まずは

  * 3D FNO（FFTベース）
  * もしくは 3D U-Net + 周波数チャネル
    から始めた方が安定しやすいです。

（PINOなど物理制約を混ぜるのは、データが少ない領域では有効になり得ますが、まずはSDF表現の安定化が先です。）

---

## 3-4. 多忠実度の学習戦略（必須）

* Stage A：LFで事前学習（run数を稼ぐ）
* Stage B：HFで微調整（全体を更新しない、アダプタのみ更新＝PEFT推奨）
* Stage C：SEMの10条件は「アダプタ」か「latent」だけ更新

---

## 3-5. 必要データ量の目安

* LF：500〜3000 run（粗い/簡略物理で高速に生成）
* HF：30〜80 run
  → これで “多形状へ展開”が現実的になります

---

# 実測（SEM）データ：必要数の目安と形式

あなたの制約（工程ごと10条件程度）を前提に、「これなら回る」仕様です。

## SEMの形式（推奨）

* 生画像：TIFF/PNG（16bit推奨）
* 付随メタ：倍率、スケール、断面方向、工程ステップ
* 前処理出力（必須）：輪郭ポリライン or マスク

  * JSON（polyline）またはPNG（binary mask）

## 実測データ数の目安

* **最低**：10条件（既にある想定）
* **望ましい**：20〜30条件あると、latent同定と探索が安定しやすい
  ただし、**サロゲート本体の学習はシミュレーション中心**にし、実測は
* latent同定
* 小さなアダプタ微調整
  に使うのが運用コスト的に正解です。

---

# レシピ探索（最適化）まで繋ぐ場合の実装

* 1位/2位/3位どれでも、最適化は基本同じです。
* BoTorchは cost-aware / multi-fidelity を含むチュートリアル・機能が整備されています ([BoTorch][5])
* HF（2時間）とLF（数分）を混ぜてMFBOで回すのが最も現実的。

---

# どれから着手すべきか（実装順のおすすめ）

* **まずは1位のMVPを作る**（SDF narrow band + (V_n) + レベルセット更新）

  * これができれば「高速で時間変化が出る」まで到達します
* 次に

  * Teacher用に内部ログを吐く（特権情報）
  * SEM特徴抽出を固める
  * latent同定（MAP→SBI）
    へ進むのが最短です。SBIはsbiで実装できます ([GitHub][4])
* 2位（表面GNN）は、1位が動いた後に「表面中心へ軽量化」「非局所を強化」したい時に追加
* 3位（Operator）は、複数形状展開の中長期課題として並行研究が向きます

---


[1]: https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html?utm_source=chatgpt.com "Sparse Tensor Networks — MinkowskiEngine 0.5.3 ..."
[2]: https://www.sciencedirect.com/science/article/pii/S2352711025004194?utm_source=chatgpt.com "ViennaPS: A flexible framework for semiconductor process ..."
[3]: https://joss.theoj.org/papers/10.21105/joss.07754.pdf?utm_source=chatgpt.com "sbi reloaded: a toolkit for simulation-based inference workflows"
[4]: https://github.com/sbi-dev/sbi?utm_source=chatgpt.com "sbi is a Python package for simulation-based inference ..."
[5]: https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/?utm_source=chatgpt.com "Cost-aware Bayesian optimization"
