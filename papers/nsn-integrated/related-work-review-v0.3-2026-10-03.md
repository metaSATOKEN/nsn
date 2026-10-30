# NSN統合稿：関連研究の拡張と採用記録 v0.3

2026-10-03。対象は[日本語v0.4](Old/manuscript-ja-v0.4-2026-10-03.md)と[英語v0.2](Old/manuscript-en-v0.2-2026-10-03.md)。[初回ギャップ調査](research/related-work/gap-scan-v0.1-2026-10-03.md)で定めた六領域を、一次資料の方法節から点検した。引用件数を満たすための追加ではなく、構成と主張に対応する比較相手を補う作業である。

## 結論と改訂範囲

ノルム／方向の分離はKV圧縮への先行適用がある。現在のNSNは、token別の有限精度ノルム保存、回転なしの方向成分のチャネル別非対称scalar量子化、復元後の再正規化という具体的な構成として説明する。その構成群、適用条件、実packed保存、読み出し・検査・初期圧縮の費用を対応づけた観測が、本統合稿の評価対象である。

参考文献は12件から26件へ。既存番号1–12は維持し、一次本文の関連方法節を確認した14件を13–26へ追加した。導入、関連研究§7と表9、考察§8、参考文献を日英同時改訂した。数値結果、分母、閾値、数式、QDQコード、図4点は維持した。追加文献の結果を、NSNの機序同定や未実施の性能比較には使わない。

これは範囲を定めた関連研究調査であり、全検索結果の系統的選別や完全な被引用探索を行ったsystematic reviewではない。「初めて」「全研究を網羅」「類似方式が存在しない」という結論には用いない。

## 六領域の対応

|領域|今回の比較対象|本文での役割|
|---|---|---|
|ノルム・方向・角度・球面|MiniCache、PolarQuant、TurboQuant、FibQuant、OCTOPUS、SPHERICAL KV|§1、§7.1／表9。一般原理と具体的構成を分ける|
|正規化・回転・量子化軸|KIVI、KVQuant、QuaRot、KVarN、Coupled Quantization|§7.2。NSNの構成比較が動かしていない要素を明示|
|K/V差・attention誤差|QJL、AsymKV、NOVA-KV、KVarN|§7.2、§8.1。誤差目的と適用条件を対応づける|
|再帰型への適用|Hyperloop、LoopQ、Looped Latent Attention、FlashLoop、Li|§7.4。重み量子化・周回間共有・各周回KVの低bit化を分ける|
|実保存と読み出し費用|KIVI、FlashAttention、FlashInfer、OCTOPUS、SPHERICAL KV、FibQuant|§7.3。一般的な融合と、今回の算術・検査・初期費用の比較を分ける|
|品質評価の範囲|KVarN、既存の各方式の評価節|§8.1–8.3。短いteacher-forcingから生成時の蓄積を外挿しない|

## 優先5件で確認した比較点

方法の要約は本文§7.1–7.2と表9に集約した。ここでは参照節と、誤読を避けるための区別を残す。

|資料・固定した参照版|確認箇所|NSNと区別する点|
|---|---|---|
|[MiniCache v2](https://arxiv.org/html/2405.14366v2)|§4.2、式2–3|層間の方向共有と、区画内の量子化は異なる。分離とノルムに合わせた復元の一般性をNSN固有としない|
|[PolarQuant v1](https://arxiv.org/html/2502.02617v1)|§2.2–3.2|角度の量子化と直交座標の成分量子化を分ける。最終半径は残るため、補助情報全体がゼロという読みを採らない|
|[TurboQuant v1](https://arxiv.org/html/2504.19874v1)|§1.3、§2–3|非単位ノルムの扱いを確認。MSE向けと内積推定向けの構成を一方式へ混ぜない|
|[KVarN v1](https://arxiv.org/html/2606.03458v1)|§2.1–3.4|分散スケーリングはtoken別L2ノルム保存と同じ処理ではない。生成評価の観測をこちらの機序実証としない|
|[FibQuant v1](https://arxiv.org/html/2605.11478v1)|§2–3.4|共通vector codebookとNSNのscalar min/maxを分ける。同論文の復元式をNSNの最終再正規化と同一視しない|

MiniCacheは初回調査のv1からv2の該当節へ照合し直した。他4件は今回取得できた履歴で上記v1を参照版として固定した。PolarQuantはHanらのarXiv:2502.02617を指し、同名資料を題名だけで結合していない。参照版の固定は、将来の改訂版や会議版まで同一であるという保証ではない。

## 追加9件の一次確認

|資料・版|確認した関連箇所|採用先|
|---|---|---|
|[QJL v1](https://arxiv.org/html/2406.03482v1)|§1.1、§3：Kのsketchと内積推定|§7.2|
|[QuaRot v1](https://arxiv.org/html/2404.00456v1)|§3.4–4：回転・重みへの吸収・量子化範囲|§7.2|
|[AsymKV v1](https://arxiv.org/html/2410.13212v1)|§3–4：K/V誤差経路、層別精度|§7.2、§8.1|
|[Coupled Quantization v1](https://arxiv.org/html/2405.03917v1)|§3.1–3.2：チャネルの結合単位|§7.2|
|[NOVA-KV v1](https://arxiv.org/html/2608.04074v1)|§2.3–4：attentionを踏まえた変換と符号化|§7.2、§8.1|
|[FlashInfer v1](https://arxiv.org/html/2501.01005v1)|§3.1–3.3：cache形式・template・runtime|§7.3|
|[OCTOPUS v1](https://arxiv.org/html/2605.21226v1)|§3.1–3.6：座標表現・量子化・Triton内復元|§7.1／表9、§7.3|
|[SPHERICAL KV v1](https://arxiv.org/html/2605.18856v1)|§2.1、図2：角度符号上のK類似度|§7.3|
|[Li v1](https://arxiv.org/html/2609.30820v1)|§3–6：重み誤差の再帰への入り方と周回校正|§7.4|

題名・著者・arXiv IDを一次ページに照合した。FlashInferのHTML先頭には組版警告があるが、本文と書誌の確認とは分けた。前回取得不成立だったLiは今回の版付きHTMLで関連本文を取得できた。表の「確認」は必要な方法節の読解であり、全定理・全結果・実装の再現を意味しない。14件の書誌、URL、節、採否は[機械可読catalog](research/related-work/source-catalog-v0.1-2026-10-03.json)にも保存する。

## 既存資料と出版上の来歴

既存1–12の資料と採用範囲は[以前の一次確認記録](Old/related-work-review-v0.2-2026-10-03.md)に残す。旧NSNの公開版は[版照合記録](publication-provenance-v0.1-2026-10-03.md)のとおりZenodo v1.0.0。既公開結果の再集計、未公開SADPの評価、Looped系列の追加評価という区別を維持する。

2026年の近接研究には旧NSN公開後のものも含まれる。「本稿に関連する研究」と「旧NSNに先行した研究」は同義ではない。掲載時刻の細かな優先関係や、他論文からの影響関係は推定しない。

## 本文へ入れなかった候補

|候補群|今回の確認水準|判断|
|---|---|---|
|[D-Quant](https://arxiv.org/abs/2609.19880)|一次要旨の検索表示。方法本文は未確認|entropy coding／ランダムアクセスの候補棚。本文の技術的主張に使わない|
|RaBitQ|FibQuant等の参考文献から発見。元本文未精読|ANN向け量子化の背景候補。KV適用の検証済み手法としては採用しない|
|Weight Normalization、DoRA|MiniCache等の参考文献から発見。元本文未精読|重み・学習側の背景であり、今回のKV比較の必須項目へ追加しない|
|GEAR、Palu|関連文献から発見。元本文未精読|低ランク等の別圧縮軸。本文で方法・性能を断定しない|
|広いsurvey・二次記事|検索の手掛かり|技術的比較の根拠は上記の一次本文に限定|

今後、低ランクやentropy codingとの性能比較を主張するなら、この棚の本文確認と比較条件の再設計が必要になる。現在の論文の完成条件へ未実施の外部比較を一括追加しない。

## 検索と終了条件

検索日は2026-10-03。[初回調査](research/related-work/gap-scan-v0.1-2026-10-03.md)の17検索語に加え、今回以下の8検索語を使い、優先論文の関連研究・参考文献を一段たどった。

- `"KV cache" "norm" "direction" quantization 2026`
- `"MiniCache" "PolarQuant" KV quantization`
- `"KVarN" "quantization" recurrent`
- `looped transformer KV quantization FlashLoop Hyperloop LoopQ`
- `KV cache quantization norm preserving normalization direction sphere`
- `KV cache quantization survey scalar vector error accumulation`
- `"MiniCache" "PolarQuant" "quantization"`
- `"Quantizing Looped Transformers"`

今回の追加検索はarXivドメインを入口とし、一次HTML／要旨へ進んだ。候補名・IDによる本文内検索も併用した。検索結果の全件数と排除数は採取していないためPRISMA型の網羅性や再現率は報告しない。完全な前方被引用索引、非英語文献、商用実装、非公開研究の探索は対象外。

六領域に比較対象を置き、優先5件と本文採用14件の関連方法節・指定版を固定し、未確認候補を分離したため、この改訂の調査をここで閉じる。追加実験、GPU実行、PDF作成、外部公開は行っていない。github-releaseは以前の検証済みsnapshotを維持し、今回の論文改訂をまだ反映していない。
