# EXECPLAN — P0/P1/P2 実行計画

## P0（スキャフォールド + 自動実行が回る）
目的：依存追加無しで、パイプライン骨格・幾何ユーティリティ・CLI/verify・autorun が動く状態を作る。
- srcレイアウト + preflight + commands SSOT
- feature/preprocess/train/infer の抽象とパイプライン合成
- SDF/level-set の簡易実装（numpy）と合成データ生成
- verify: import + unit + smoke
- checkpoint（P0末尾）

## P1（最有力手法のMVP）
目的：有力順1位の構成を「最小で end-to-end」動かす。
- SDF narrow band データスキーマ（HDF5）確定
- Vn教師（SDF差分/grad）生成
- モデルはまず軽量（numpy/torch optional）で Vn を回帰
- rollout loss（短窓）でロールアウト崩壊を防止
- SEM特徴抽出のI/O（輪郭→特徴ベクトル）
- latent z のMAP校正（モデル本体固定）
- inference: OODフック（距離ベース）
- decision: 依存追加（torch等）を有効化するか

## P2（拡張・研究開発）
目的：性能/汎化/同定/探索を強化し、運用コストを下げる。
- Teacher-Student（特権情報：フラックス統計など）蒸留
- 表面GNN/GraphTransformer（非局所結合：遮蔽/反射）
- Neural Operator（φ0,c,t→φ(t)）と多忠実度
- SBI（posterior）による latent 推定（sbi等）
- BO/MFBO（レシピ探索）と評価サマリー
