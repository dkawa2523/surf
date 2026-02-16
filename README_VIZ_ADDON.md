# VIZ Add-on (Post P0) — 追加可視化/出力機能パック

このzipは、**既存リポジトリ（P0-CHECKPOINT完了後）**に対して、以下の機能を後付けするための「タスク＋関連ファイル」パックです。

- 時間ごとの出力を **VTI (.vti) + PVD (.pvd)** として保存（ParaViewで時系列可視化）
- SDF/表面処理後の3Dデータを **PNGクイックルック（断面スライス）** として保存
- validation/test で **横から見た断面輪郭（pred vs gt）比較画像** を保存
- 学習/評価の **グラフ（loss/metrics）** を保存

---

## 使い方（推奨）

1) **このzipをリポジトリrootに展開**（既存ファイルは上書きしません。`scripts/addons/` に追加されます）
2) preflight（未実行なら）
   ```bash
   ./scripts/preflight.sh
   ```
3) add-on適用（tasksの追加 & docs追記はオプション）
   ```bash
   ./scripts/addons/apply_viz_addon.sh
   ```
4) codex autorun 再開（続きから）
   ```bash
   ./scripts/run.sh
   ```

---

## 何が起きるか
- `tasks/tasks.json` に **P0-VIZ-001〜004** が追加されます（デフォルトは `P0-CHECKPOINT` の直後、`DECISION-P1-001` より前に挿入）。
- 実装が完了すると `runs/<run_id>/viz/` 以下に VTI/PNG/plots が生成されます（詳細は `docs/VIZ_SPEC.md`）。

---

## 注意
- 画像/グラフ生成は `matplotlib` があると充実します。無い場合は graceful fallback（VTIのみ等）を実装する方針です。
- 生成物は `runs/` に出るため git 管理しません。
