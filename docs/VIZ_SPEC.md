# VIZ_SPEC — 可視化/出力拡張（VTI・断面輪郭・グラフ）

このドキュメントは「形状サロゲート（SDF/level-set）実装」に対して、**実装後に追加する可視化・出力機能**の仕様（I/O契約）を定義します。
Codexタスク（P0-VIZ-***）は本仕様を参照して実装します。

> 目的
> - 時間ごとの形状（SDF φ, occupancy, label など）を **VTI（VTK ImageData）** として出力し、ParaView等で時系列可視化できるようにする
> - SDF/表面処理後の3Dデータを「軽量なクイックルック画像（PNG）」で確認できるようにする
> - 学習評価（validation/test）で **横から見た断面輪郭**（pred vs gt）比較画像を出力する
> - 学習/評価の指標を **グラフ** として出力し、第三者が run を理解できるようにする

---

## 1. 用語

- **φ (SDF)**: Signed Distance Field。φ=0 が界面（表面）。
- **narrow band**: |φ|<w の近傍領域。学習や更新で用いる。
- **run_dir**: `runs/<run_id>/` を指す。

---

## 2. 出力ディレクトリ（run_dir配下）

可視化生成物は以下に集約する（git管理しない想定）。

```
runs/<run_id>/
  viz/
    vti/                 # VTI & PVD（時系列）
      phi/               # φ(t)
        phi_t0000.vti
        phi_t0001.vti
        ...
        phi.pvd          # time series collection
      occ/               # occupancy(t) (optional)
      label/             # material/mask label (optional)
    png/
      slices/            # 3Dクイックルック（断面スライス）
        slices_t0000.png
        ...
      compare/
        val_sample000_y032.png
        test_sample003_y032.png
    plots/
      train_loss.png
      val_metrics.png
      metric_hist.png
    summary/
      viz_manifest.json  # 生成物一覧（optional）
```

---

## 3. VTI（.vti）出力仕様

### 3.1 最低限の互換性
- VTIは VTK XML ImageData 形式（拡張子 `.vti`）。
- **最初は ASCII 形式でよい**（ファイルが大きくなるが依存なしで実装でき、ParaViewで読める）。
- 大規模データ向けに将来 `vtk`/`pyvista` 依存を許可する場合は binary/appended へ拡張可（今回のタスクは必須ではない）。

### 3.2 形状・並び順（重要）
- 配列 shape は **(nx, ny, nz)** を基本とする。
- VTKのImageDataのスカラー並びは i(x) が最内側で変化するのが自然なため、
  **flatten は `order='F'`（Fortran order）** を標準とする。
- extent は `0 nx-1  0 ny-1  0 nz-1`

### 3.3 時系列（PVD）
- ParaViewで時間シークできるよう、`phi.pvd` を出力する。
- `phi.pvd` は各 `phi_tXXXX.vti` と time 値を列挙する。

---

## 4. クイックルック画像（PNG）

### 4.1 スライス画像（slices）
- 1枚のPNGに **XY / XZ / YZ** の3断面を並べて表示（もしくは縦に3枚）。
- 可能なら φ=0 の輪郭をオーバーレイする。
- 断面位置はデフォルトで中心（y=ny//2 等）とし、CLIで指定可能にする。

---

## 5. 断面輪郭比較（pred vs gt）

### 5.1 断面の定義（横から見た比較）
- デフォルト断面は **XZ面（y固定）** とする（“横から”に相当）。
- `y_index` を指定できるようにする（デフォルト `ny//2`）。

### 5.2 出力画像
- 1画像に pred と gt の **φ=0 輪郭**を重ねる（色/線種で区別）。
- 可能なら差分領域（inside/outsideの不一致）も薄く可視化する（optional）。

---

## 6. グラフ（plots）

- 学習ログ（jsonl/csv等）から以下を生成:
  - train loss curve
  - val metrics curve（複数）
  - metrics histogram（val/test）
- 入力ログ形式は多様でよいが、実装側は run_dir 内から既知候補を探索する：
  - `metrics.jsonl`, `train_metrics.jsonl`, `metrics.csv`, `eval_metrics.json` など

---

## 7. CLI（想定）

`./scripts/commands.sh run -- viz <subcommand> ...`

例：
- VTI出力（スモーク）  
  `./scripts/commands.sh run -- viz export-vti --smoke`
- スライス画像生成（run_dir指定）  
  `./scripts/commands.sh run -- viz render-slices --run-dir runs/<run_id>`
- 断面比較  
  `./scripts/commands.sh run -- viz compare-sections --run-dir runs/<run_id> --split val`
- メトリクスプロット  
  `./scripts/commands.sh run -- viz plot-metrics --run-dir runs/<run_id>`

---

## 8. 非機能要件
- 生成物は `runs/` に集約する
- 既存のP0機能を壊さない（互換性維持）
- 依存追加なしで動く（matplotlib等は存在すれば利用、無ければ graceful fallback も可）
