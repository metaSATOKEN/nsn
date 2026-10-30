# LOOP-KV-KV01：Kのみ／Vのみの局所圧縮比較 v0.1

2026-09-30、Kentaro Sato。取得前固定。P1B後の探索的な局所介入であり、未使用材料による確認試験ではない。ユーザーが小比較の設計と実装まで承認した。本機ではモデルをロードせず、実測はColabで行う。

## 1. 問いと範囲

P1Bではr8・NSN3・K＋Vの相対PPL悪化は約0.644%だった。次は同じ8冊で、KとVのどちらを圧縮した時に損失が大きいか、両方の損失が個別損失の和で説明できるかを測る。P1Bの結果を見てr8・3bitを選んだことを明記する。

K/Vは全8周回・24層・16 headをまとめて扱う。周回別重要度、同じメモリ予算での最適配分、改善の有無を今回の成功条件にはしない。Kのみ／VのみはK＋Vの半分の要素を量子化する。方向の事前予想は置かず、探索的な対称比較とする。加算性についても対称的に三分岐させ、結果を見る前に幅を固定する。

## 2. 32軌道と条件

| 条件 | K | V | 軌道数 |
|---|---|---|---:|
| NATIVE | BF16 | BF16 | 8 |
| K_ONLY | NSN3→BF16 | BF16 | 8 |
| V_ONLY | BF16 | NSN3→BF16 | 8 |
| KV | NSN3→BF16 | NSN3→BF16 | 8 |

同じ8冊3752, 31131, 31258, 34475, 4124, 49966, 49484, 4537。材料・tokenizer・offsetはP1Bと同一。各条件1軌道、32軌道×255予測＝8,160 token評価、8,192 forward。独立材料は8冊。token、head、周回を独立標本として水増ししない。

r=8、PREFIX_ONLY、prefix768 tokenのうち先頭704 tokenを64-token blockで一度だけ量子化。直近64 tokenと後続255 tokenはnative。KはRoPE後、NSN3の方向量子化とFP16 norm/min/step、再正規化は凍結codecのまま。重み量子化、clipping、再学習は行わない。

全軌道はfresh cache。書籍順と各冊の介入3条件の順をseed2026093001で固定し、各冊NATIVEを先に実行する。NATIVEとKVも今回新規に測り、過去の値をbaselineへ流用しない。過去実験を反復として合算しない。target位置769〜1023の同じ255点でteacher-forced NLL/PPLとnative→条件KLを測る。

## 3. 実行環境

ByteDance/Ouro-1.4B、revision fcfcfbc429c7b10ee73dfc718033351d8936d2a8。BF16、SDPA AUTO、TF32 off、GEMM reduced BF16 on、math SDPA reduced off。固定依存版とremote code hash、tokenizer hashを照合し、不一致時は停止する。rはconfigとmodelの3属性を同期、exit_at_step=r−1、labels=None、use_weighted_exit=False。

実装はfake quantizationで、cacheはBF16へ戻す。実測float cache bytesと、side別のcode・metadata・native tailを数えた理論packed bytesを分ける。実VRAM削減・高速化は今回の測定対象ではない。

## 4. 本比較前の進行ゲート

旧BF16 full/incremental P0未達は変更しない。今回のr8に必要な同じ逐次形状の一致を新たに点検する。

- G2：元smoke4冊×r8、長さ129／prefill64。4 case、1,060 forward。
- G3：合成境界長1/63/64/65/127/128/129×r8。7 case、807 forward。
- G4：元smoke4冊×r8、長さ1023／prefill768。4 case、4,096 forward。

計15 case、公式再実行／新adapter／新adapter＋traceの45比較、5,963 forward。logitsと最終全KVのbitwise一致、RNG不変、285,696 cache更新座標を検査。full/incrementalの11診断は元閾値をそのまま記録し、進行ゲートと分ける。

追加G5：同じsmoke4冊のr8・NSN3・PREFIX_ONLYについて、凍結された従来のK＋V adapterと新しいKV条件を比較する。各々fresh cache、255予測＋prefillで2,048 forward。全生成logitsと最終全KVのbitwise一致、RNG不変を4比較で必須化する。新側は介入直後検査を有効にし、計器を含めて一致を確かめる。

合計19 case・49比較・8,011 forward。固定codec fixtureと新しいside選択fixtureを含め、全合格時だけPASS_ROUTE_KV01で32軌道へ進む。数値不良や不一致はBLOCKED_ROUTE、時間・OOM等の未完了はINCONCLUSIVE。G5は過去GPU結果との一致ではなく、今回のGPU上で旧実装と新実装を比較する。

## 5. 介入の監査

本比較の各軌道のprefix直後に、全192 slot×K/V＝384 sliceを検査する。選択した側の704 tokenだけをcodecへ渡し、未選択の側全体と両側の保護tail64 tokenのbitwise不変を必須とする。NATIVEは全KV不変。選択側では変化ゼロの正常入力も許容する。

検査は介入前snapshotと介入直後を比較する。後続推論で未選択側の値が変わることは自然に起こり得るため、最終cacheの未選択側をnativeと同じにする要求は置かない。slice別に選択の有無、検査範囲、最大絶対変化、相対L2変化、保護範囲の一致を保存する。完了時は12,288 slice記録。圧縮済token/block数と理論bytesはK/V別に保存する。

norm計器は各周回の層0/12/23、K/V、cache長768/832/896/960/1023。32軌道で7,680 snapshot、1,536初回transferを予定。target896では全語彙分布とnative参照をNPZで保存。その他はtoken別値と分布hash、最終cache hashを残す。

## 6. 固定解析

主比較D：各冊で NLL(K_ONLY) − NLL(V_ONLY)。8冊平均が+0.005以上ならK_LARGER、−0.005以下ならV_LARGER、その間はWITHIN_BAND。正符号はK圧縮時の損失が大きいことを表す。±0.005 nats/tokenは探索上の記述幅であり、有意差／同等性の判定ではない。

副比較I：NLL(KV) − NLL(K_ONLY) − NLL(V_ONLY) ＋ NLL(NATIVE)。8冊平均が+0.005以上ならSUPERADDITIVE、−0.005以下ならSUBADDITIVE、その間はWITHIN_BAND。NLL尺度上の非加算性であり、物理的共鳴・機構の証明とはしない。

両比較の冊別値と平均、同じ冊を再標本化する10,000回bootstrap（seed2026093002）の名目95%区間を併記する。8冊が全部そろわなければ固定判定はNOT_EVALUABLE、利用可能分の平均と区間は条件付き参考値として残す。DにはK/V両条件、Iには4条件が必要。

品質参考：各介入−今回同冊NATIVEの平均ΔNLL、exp(平均ΔNLL)−1の相対PPL、冊別値・最大悪化。P1Bと同じ5%目安：8冊そろい区間上限≤log(1.05)ならWITHIN_5PCT、下限>log(1.05)ならEXCEEDS_5PCT、それ以外UNCERTAIN。欠測時NOT_EVALUABLE。3比較の区間は同時区間ではなく探索的記述。品質向上を合格条件にしない。

予定32件の完遂／数値失敗／未実施と、D/I/品質の向きは別の欄に保存する。全件有限でCOMPLETEでも、品質悪化や大きな差があり得る。結果の符号にかかわらずこの範囲で閉じ、条件追加・再試行は行わない。

## 7. 運用・保存・上限

Colabのfresh GPU runtime、HF_TOKEN Secret、Drive接続なし、過去ZIPアップロードなし。モデルと必要な固定依存だけ取得し、研究ログを外部へ送信しない。準備30分＋進行ゲート30分＋比較60分＝最大2時間、別途終了・ZIP回収猶予。1 gate case180秒、1本比較900秒、保存10GiB、追加の課金操作なし。実際の時間はGPU次第で、上限は所要時間の予測ではない。

本比較のNUMERICAL_FAILUREは原ログを保存して次のfresh条件へ進む。nativeが欠けた場合も他条件のNLLは取得し、native参照KLと該当比較は欠測。装置不整合・OOM・期限・容量超過は停止。自動リトライ・都合の良い応答選択・パラメータ変更なし。gate不良時は本比較へ進まない。

各case／軌道後にcheckpoint ZIPを更新、終了・例外時に最終ZIPの自動ダウンロードを要求する。ブラウザのDL制限やランタイム消失は保証できないためFiles欄にもZIPを残す。tokenはログで秘匿し、環境変数全体・Drive・モデル重みをZIPへ含めない。保存期限未設定。原回答と過去設計は上書きしない。

## 8. 閉じ方と保留棚

Colabから返ったZIPをhash照合し、token別原値からD/I/品質を再計算して閉じる。今後、微小差の再現性を主張する前に未使用材料と反復を別設計する。長文脈／ONLINE、周回別介入、固定予算配分、LAB型clipping、実packed cacheは今回含めない。新しい具体的問いが生じた時に別カードを作る。


## 接続準備記録（取得前）

- worker SHA256：`7f248fce97af1c6724e555d07b8f9fae3ccf181947663a9b0517d24e3284ef11`
- 1セル SHA256：`352d244faf52191abda31511f7e8cd08cf97285871afe212e3446b6c9249b094`
- セル内設計原本 SHA256：`25bd7d063e8110a3066e119590ec2e245fc3538e6fadb4dddd326ca60a544a12`
- セル内取得前カード SHA256：`2dbcc1f894cab08f19a196880250e3490bb022b814b1c528426b7a07e43b207c`
- 固定材料 SHA256：`739a619d7486f34a8ab11c494702e804803ac170e45563fd7abde16bbcc0cc6f`

CPU合成・模擬Colab65検査合格。配布セルをbuilderから再生成してbyte一致を確認した。手元でのモデルロード・GPU実測・外部送信は行っていない。設計の原本はコード作成前に保存し、上記hashは配布準備の追記である。実行手順は[Colab手順書](../colab/kv01_README.md)。
