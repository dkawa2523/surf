# VIZ_REQUIREMENTS_ADDON — 可視化/出力拡張（追加要件）

このファイルは `docs/REQUIREMENTS.md` へ追記するための追加要件（提案）です。
運用上は `docs/REQUIREMENTS.md` に統合するのが望ましいですが、衝突回避のため分離して提供します。

## Should（推奨）

### R-SHOULD-VIZ-001 時系列VTI出力
推論・シミュレーションの時間発展（φ/occ/label 等）を **VTI（.vti）** で出力でき、ParaView等で可視化できること（PVDで時系列もサポート）。

### R-SHOULD-VIZ-002 3Dクイックルック画像
SDF/表面処理後の3Dデータについて、スライス画像（XY/XZ/YZ）などの **クイックルック（PNG）** を出力できること。

### R-SHOULD-VIZ-003 断面輪郭比較（val/test）
Validation/Test に対して、pred vs gt の横断面輪郭（φ=0）を重ねた比較画像を出力できること。

### R-SHOULD-VIZ-004 学習・評価グラフ出力
学習ログと評価結果について、loss/metrics の推移や分布をグラフとして出力できること。

### R-SHOULD-VIZ-005 再生成可能なCLI
run_dir を指定して、VTI/画像/グラフを **再生成**できるCLIを提供すること（再学習不要）。
