# NB06 準備報告

2026-10-08 / v0.1。設計・実装・CPU/模擬検証まで完了。新しいモデル/GPU実測は0件。

**NB05の初期入力を保ち、継続256 step・追加圧縮4回を測る1セルを準備した。**

[全文1セル](../colab/loop_kv_nb06_onecell.py) / [同内容ipynb](../colab/loop_kv_nb06_onecell.ipynb) / [手順](../colab/nb06_README.md) / [固定設計](../plans/loop-kv-nb06-design-v0.1-2026-10-08.md)

## 比較の範囲

同じ4冊・先頭128 token、NATIVE / PACKED_SDPA / GQA_FUSED、各3技術反復。補助の公式一致・実Q診断・全長warmupを含め60系列・15,420 forward。各系列は新cache。8/64/128/256 stepで累積時間、全256と64-token区間でPPLを示す。

追加圧縮の時間と一時メモリも計上する。NATIVEにも同じforward/post計時境界を置き、postは無処理の対照とする。初期費用込みの利益が出なくても固定256 stepで閉じる。終端4回目の圧縮は後続attentionへまだ使われていないことを結果へ明示する。

## 成功版の再利用と差分

基準は実GPUで完走したNB05返却ソースと保存manifest。参照コードをhash照合し、原ファイルを変更せずNB06へコピーした。

- **11モジュールは識別子置換のみ：** NSN算術、codec、packed cache、GQA kernel、入力検査、attention adapter、実Q捕捉、数値検証、診断、storage/evidence、専用runtime。
- 変更：固定4文章の385-token延長と旧137-token一致、256-step制御、post_decode計時、複数block不変性の記録、系列別JSONL、固定8位置のraw logits、全位置のNLL成分、長さ別再集計、回収予定数と上限。
- 新しい量子化・復元算術やkernel tuningは加えていない。既存cacheの追加圧縮をNanbeige長系列へ接続する部分は、新たなGPU確認対象。

[再利用元hashと差分](../throughline/audits/loop-kv-nb06-code-v0.1-2026-10-08/reuse-and-diffs.json)を保存した。

## 検証したこと

CPUと模擬条件の**106項目が合格**。単なるsyntax検査に加え、CPU tensorを使うモデル代替で1文章15系列・3,855 forwardを通し、全256 stepと4回の追加圧縮を実際の測定制御/保存器で走らせた。独立NumPy検証器が保存物を再集計した。4文章全体の検証器は、このCPU記録に合成ID・時刻・token bindingを与えたfixtureで検査した。これは実モデルの15,420 forwardではない。

- コピー元の公式projection/norm/RoPE/SDPAに対するadapter照合。
- 全slotのstorage、headerとappend位置、既存blockの不変性、入力/出力検査回数。
- step欠落、損失/成分改変、圧縮費用欠落、更新/帰属誤り、captureの主計時混入、sparse位置の改変を検出。
- 技術反復の不安定はINCONCLUSIVE、有限の保存/経路不一致はFAILとして保持。JIT cache変更は接続成績と速度解釈を分離。
- 固定材料の本文hash/先頭token不一致や取得失敗で、勝手に別の本を採用しない。
- 別venvへの実process経由のオフラインpip fixture導入、GPU loader環境の継承、Secret非露出、途中停止/偽完了/認証失敗/ブラウザDL拒否からのZIP回収。
- .pyと.ipynbの1セル内容一致、29埋込ファイルのhash、固定契約の同梱。

試験途中、欠けたpost計時の故障fixtureをTypeErrorで拒否していた箇所に、明示的な構造検査を加えた。全長比較ではNATIVEにも同じpost計時境界を置き、無処理区間のCUDA時間0を許容するよう取得前に固定した。GPU結果や既存成績に合わせた変更ではない。

## 証拠・運用上限

全位置のNLL成分とlogit hash、非warmup48系列×固定8位置の生logits（243.375 MiB）、実Q32点、全step時間/storageを保存する。非保存位置は原logitsから後日再計算できないため、成分再計算と反復hashの検査として区別する。

Drive不要・前回ZIP不要・HF_TOKEN・自動DL。準備30/測定60/照合10/ZIP5分、全体105分は停止上限で所要時間予測ではない。証拠768 MiB・ZIP512 MiB。初期ディスク30 GiB、重み約7.77 GiB。追加有料APIなし、Colab利用枠を消費。

GPUでの数値整合・速度・品質、Colabの現在環境との適合は返却ZIPで確認する。手元モデルロード・外部投稿・校了稿の改訂は行っていない。
