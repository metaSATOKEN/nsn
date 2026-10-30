# LOOP-KV-NB06 — Nanbeigeで圧縮KVを更新しながら256 step使う

2026-10-08 / v0.1。ユーザーは継続長を延ばす小比較の設計とColabコード化を一括承認した。新GPU結果は未取得。暫定校了した統合稿とNB05原結果は保持する。

## 1. 問いと固定する条件

**短い接続試験で成立した実packed NSN3を使い続けたとき、品質・保存量・使用費用はどう変わるか。** 今回は最初に読む文脈を長くせず、その後の継続を8から256 stepへ延ばす。長い自由生成や長文検索能力は別の問いとして残す。

成功run `loop-kv-nb05-20261003-090604-18baabf5` のコードをコピーし、差分で実装する。モデル、復元・融合算術、数値予算、入力検査、runtime導入・認証・回収は継承する。

| 条件 | 固定値 |
|---|---|
| モデル | Nanbeige/Nanbeige4.2-3B-Base、revision `4a38e817c14b9f2c4b69b1e05cbda90395872f6b` |
| 形状 | BF16、batch 1、22層×2周=44 slot、Q48/KV8、head次元128 |
| 初期入力と材料 | NB05の4冊（9667、26438、45277、28881）、先頭128 token |
| 全入力 | 各385 token、同じ正解tokenを使うteacher forcing |
| 継続 | 256 forward、主target位置129〜384。prefill target128は主品質から除く |
| 保存 | NSN3、64-token block、norm/min/scaleはFP16、末尾64〜127 tokenをBF16保持 |
| 比較 | 検査付きNATIVE／PACKED_SDPA（全量復元）／GQA_FUSED（内部復元） |

本文をColabで再取得し、NB05の全文hashと先頭137 tokenを確認してから385 tokenへ延長する。本文差替え、別の本への置換、結果を見た材料選定はしない。tokenizerは同じ固定revision、use_fast=False / legacy=True / add_special_tokens=False。全tokenと予定を重み取得・モデルロード前に凍結する。各本文16 MiB・90秒、4件・合計64 MiB、再試行0。全文は返却せず、hash・出典・使用tokenを保存する。

## 2. 追加圧縮を繰り返す

既存cacheの `post_decode()` を利用する。処理は**追記→attention→そのforwardの結果取得→post_decodeで圧縮**。過去のpacked blockは再量子化せず、新たな64 tokenを末尾から移す。

| 処理後 | 全長 | 圧縮済み | BF16末尾 | 備考 |
|---|---:|---:|---:|---|
| prefill |128|64|64|初期圧縮 |
| 8 step |136|64|72|前回と同じ長さ |
| 64 step |192|128|64|1回目の追加圧縮 |
| 128 step |256|192|64|2回目 |
| 192 step |320|256|64|3回目 |
| 256 step |384|320|64|4回目・終端 |

第64 stepのattentionは圧縮64＋末尾128で計算し、その後に圧縮128＋末尾64になる。更新後の状態は第65 stepから使用する。終端256 step後の4回目は保存まで測り、後続attentionでの使用は今回測らない（使用まで観測する更新は3回）。終端の圧縮費用も省略しない。

初期と各追加圧縮の前後で、全slot・K/Vの既存block（符号・補助情報・header）のhashが不変であること、新blockの開始位置、block数、末尾長を保存する。hash採取は主計時外。全forward後の実storage棚卸しで、隠れた全量BF16保持とviewによる過大保持を検出する。

## 3. 予定数・順序・warmup

各文章で、公式/NATIVEゲート2系列、融合診断1系列、各方式全長warmup3系列、主測定3方式×3技術反復9系列。**4文章×15系列×257 forward＝15,420 forward**。1系列はprefill1＋decode256。モデルロードは1回、各系列は新cacheと固定seedで開始する。

主方式順はNATIVE→PACKED_SDPA→GQA_FUSED、PACKED_SDPA→GQA_FUSED→NATIVE、GQA_FUSED→NATIVE→PACKED_SDPAの3巡。前回同様、全長warmup後に主測定へ進む。全length/圧縮block数のJITを実経路で準備し、主測定前後のTriton cache変更は速度の準備不足として別表示する。

診断系列でstep 1/65/193/256 × slot 0/43を捕捉（8点/文章、計32点）。実Q・同じpacked K/Vを三復元経路へ渡し、既存の工程別予算で照合する。主測定に診断処理を混ぜない。局所数値検査の範囲はこの固定点であり、全位置の原KVを返却する設計ではない。

## 4. 品質・時間・メモリ

### 品質

主対象は全256予測。補助窓は最初8、1〜64、65〜128、129〜192、193〜256。各文章・各方式の全窓を表示し、全体だけで終盤の不備を覆わない。窓は同じ軌道の部分集合であり独立Nではない。

反復ごとのmean NLLと同方式logit hashの一致を確認し、各文章の3反復中央値を得る。4文章のmean NLLを一度ずつ平均してPPLを計算する。相対PPLは `100*(exp(mean_NLL_arm-mean_NLL_reference)-1)`。5%は従来の診断目安で、接続合否・統計的非劣性と分ける。旧NB05値とのプール、CI、p値は使わない。同一run内の8-step値を短区間対照とする。

### 時間

全stepでモデルforwardとpost_decodeを別計時し、**decode合計は両方を足す**。累積8/64/128/256 stepで、decodeのみ、prefill＋初期圧縮＋decodeの両方を報告する。GPU eventとwallを併記。非圧縮も同じforward/post計時境界を通し、postは何もしない対照とする。postのCUDA時間は0を許容する。入力作成、CPU保存、原値hash、診断は計時外。実行に必要な入力検査とcache追記・再配置・圧縮は計時内。

各文章で3反復の対応比と中央値、全体は4文章の比率中央値の中央値。更新step64/128/192/256のforward/post費用も個別表示する。固定4チェックポイントで累積比が1を下回るかを報告し、未測定長へ償却点を外挿しない。反転しても最初に1未満になった点だけで恒久的高速化とはしない。

### メモリ

各stepのKV実storageとallocated/reservedのbefore/peakを記録。累積peakはforwardに加えて初期/追加圧縮を含む。終端予定値はNATIVE69,206,016 byte、packed24,642,816 byte（コード10,813,440＋補助2,252,800＋header42,240＋末尾11,534,336）。検証器でモデル形状とblock数から再計算する。GPU全体のpeakとKV保存量を別に表示する。

## 5. 原値と小容量回収

全15,420位置で、target、NLL、FP64のmax_logit/sum_exp_shifted/target_logit、全FP32logit vectorのhashを残す。NLLは保存成分から別実装で再計算する。全語彙logitsは非warmup48系列の**step0/1/8/64/65/129/193/256のみ**保存し、hashと成分とNLLを照合する（384行、243.375 MiB）。それ以外の位置は原logitsから後日再計算することはできず、成分再計算と反復hashで点検する。全位置raw保存と同等の検証とは記述しない。

step記録は系列別JSONLへ逐次保存する。証拠768 MiB、ZIP512 MiBが上限。モデル重み・venv・HF/Triton cache・書籍全文はZIPに含めない。停止時も原記録、未取得の予定、理由を保存して回収する。

## 6. 取得前予想と終了

- H1：4文章で公式/NATIVEの全logit hash・終端KV一致、全slotの経路/storage、追加blockの保持、固定実Q数値予算、同方式3反復安定が成立する。
- H2：融合対NATIVEの全256と終盤64のPPL悪化が全4文章で5%以内。他窓と全量復元も全表示する。
- H3：256 stepの融合/全量復元CUDA decode累積時間比（post_decode込み）の集約が1未満。
- 非圧縮に対する償却は方向を予想せず、固定4点で記述する。

PASS_MODEL_CONNECTION／FAIL_MODEL_CONNECTION／INCONCLUSIVEはNB05を継承。有限の局所予算超過、保存・経路検査の不一致、品質悪化、速度低下は予定を続けて不成立を残す。公式ゲート不一致、非有限、構造不良、証拠破損、時間/容量超過は未確定として停止。反復不安定は全結果取得後にINCONCLUSIVEとする。結果を見てbit数・長さ・kernel・材料・閾値を調整しない。

## 7. 実行・準備検証

Drive不要、前回ZIP不要、HF_TOKEN、1セル貼付、終了ZIP自動DL。手元でモデルをロードしない。NB05のCPython3.13/Linux x86_64専用venv、torch2.11.0+cu128、Triton3.6.0、transformers4.45.1を再利用する。BF16対応GPU・初期空き30 GiB。Colab利用枠を使い追加有料APIなし。

準備30分、測定60分、照合10分、ZIP5分、**全体105分が停止上限**（所要時間予測ではない）。1文章25分、1系列10分も適用。初回に多数の長さのJIT準備が入るため、最初の文章は待ち時間が長くなり得る。

配布前に、成功コードの差分、CPUでの複数block・header・prefix不変性、公式attention adapter、全長駆動の模擬実行と独立検証器、NLL改ざん・更新漏れ・費用欠落・取り違え、ZIP回収、source固定を確認する。GPUの正確性・速度は返却後に評価する。
