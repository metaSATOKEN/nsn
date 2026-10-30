# IV01：異常検出範囲を保った入力検査の集約

2026-10-01 / v0.1。ユーザーが設計からColabコードまで承認。DC01で単独入力検査0.193msが見えたため、同じ入力検査をまとめる部品比較を行う。モデル／PPL／長さ変更／V復元の融合は含めない。

## 問い・対照・介入

DC01の18実標本を再利用する。query、K tail、Kのnorm/min/scale、復元Vの六つの浮動小数配列について、有限性を検査する。型・形・配置・deviceの既存検査も毎回維持する。

| 方式 | 処理 |
|---|---|
| LEGACY | 元QK検査と元V検査をそのまま順に呼ぶ。hostによる有限性判定は2回 |
| TORCH_GROUPED | 同じ構造検査の後、六配列のtorch有限性結果をまとめてhost判定1回 |
| TRITON_GROUPED | 同じ構造検査の後、六配列の有限性を1つの新Triton kernelへまとめ、flag集約をhost判定1回 |

Vの期待長にはAC03接続口と同じK全長を渡す。DC01の合法18標本では保存V長と一致する。

構造検査は凍結関数のASTから有限性判定の節だけを除いて抽出し、元の関数は変更しない。すべての方式で、異常を検出した呼出しからattentionへ進まない。DC01のように20回分を後から確認する条件は置かない。複数の不備が重なる入力でのエラー文言・検出順は比較対象外。ヘッダー内容や出典等、元の検査器にない検査を追加したとは主張しない。

## 固定材料と比較単位

元ZIP `loop-kv-dc01-20261001-134726-c5d0df08.zip`、SHA256 `2a7de7529634f8496c76f36bfd87be72feb6413cded4be82196a4ca746ccd2fe` を使用。実K/V・queryを作り直さず、paired_inputsの18NPZをhash照合して読む。返却ZIPのPythonを実行しない。

既知2文章×長さ769/831/833×slot0/95/191。BF16のquery/tail/V、FP16のmetadata、3bit packed保存、既存RF01復元とAF01融合を固定する。3方式×2範囲（CHECK_ONLY、PACKED_PIPELINE）＝108組合せ。後者はV復元→対象検査→元融合→出力有限性確認まで含める。

主比較はTRITON_GROUPED/LEGACYのCHECK_ONLY、実経路への残り方はPACKED_PIPELINEで確認。TORCH_GROUPEDは、確認回数の集約と有限性演算の集約を分けて読むための中間条件。全18標本を示し、3round中央値→標本ごとの対応比→18標本中央値。独立Nを増やさずp値・CIなし。

## 先行する正常・異常検査

正常8fixture（ゼロ、定数、符号混在、有限最大値、subnormal、tailのみの短長2条件、packedの端数長）。異常は六配列×NaN/＋Inf/−Inf×先頭/末尾＝36条件と、型・形・非連続・device・空列・metadata構造の20条件。合計64fixture×3方式＝192判定。

保存するのは入力配列、実dtype/shape/contiguous/device、各方式の受理/拒否・例外種別、後続処理へ進んだ回数。NumPy側の独立した構造・有限性検査と比較する。正常は全方式が受理、異常は全方式がattentionへ進む前に拒否。構造異常は新Triton検査kernelの起動前に拒否する。候補が検出対象を落とした場合はINCONCLUSIVEで停止し、成績を補完しない。

次に実18標本のCHECK_ONLY受理、PACKED_PIPELINE出力が元DC01の実forward出力とbit一致することを全方式で確認する。検査は入力を変更しない。

## 計時・保存・数値条件

新kernelは固定1024要素tile・4warps・autotuneなし。各配列をtileへ分割し、連結したprogram番号で担当配列を選ぶ。領域外はmask付きloadのother=0とし、全programが各flagを上書きする。torch側でflagを集約して、確認が終わってから戻る。

全108組合せで初回1呼出しを保存し、各5warmupをすべて終えてから主計時。3round×各20呼出し＝324区間・6,480呼出し。方式順は標本とroundで循環。CUDA Event、同期wall、allocated peak、準備前後のTriton cache、入力前後hash、初回と各round終端出力を保存する。計時に有限性判定のhost確認も含める。

新GPUの数値演算を有利に変えない。torch2.11.0+cu128、Triton3.6.0、NumPy2.1.3を固定。matmul/TF32等も元と同じ条件にする。kernelを含むGPUコンパイルと性能はColabで初めて確認する。

## 判定・予想・上限

COMPLETE_COMPARISONは、全fixture・実入力gate・入力不変・反復一致・予定集合・保存照合が成立した状態。減速も完了結果として保持。主計時cache変化は原時間を残して別フラグ、結果を見た再warmupはしない。構造欠損・非有限・受理/拒否不一致・出力不一致・上限超過は途中保存付きINCONCLUSIVE。

予想：fixture判定と実出力が一致し、TRITON_GROUPEDはLEGACYよりCHECK_ONLYで短時間、PACKED_PIPELINEでも短縮が残る。TORCH_GROUPEDの改善有無、効果量、非量子化を超える速度は未予測。正常有限値であることを一度確認したからといって、後の呼出しで検査を省かない。

Colab GPU1セル、Drive・HF_TOKEN・モデルロード不要。前回DC01 ZIPを/contentへ置く。なければファイル選択を開く。準備10分＋検査/計時10分、各条件120秒、worker照合・別process照合・ZIP回収各120秒、保存512MiB。上限であり実行時間予想ではない。モデル接続は部品比較を読んでから別設計とする。

実装のmask付きloadとprogram番号の扱いは、[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)・[program_id](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html)の公式仕様を参照した。
