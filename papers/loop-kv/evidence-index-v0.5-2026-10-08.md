# 補足資料S1：本文から証拠への索引

2026-10-08 / 日本語S1 v0.5。[日本語本文v0.5](manuscript-ja-v0.5-2026-10-08.md)に対応。初回二モデル比較の証拠を保持し、§12にNB06の継続評価を追加。

公開用案内更新：2026年10月29日。利用条件・著者申告・参照先を現行パッケージへ同期した。測定値と科学的な結論は元の版から変更していない。

本索引は公開候補に収録した保存物への入口である。コード・集計・読めるログをリポジトリに、大きな保存配列を別配布ZIPに収録する（[収録範囲](../../DATA_COVERAGE.md)）。取得前計画と原記録を保持し、本文用の再表示をそれらへ上書きしない。リポジトリは[https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn)。公開範囲と別配布ZIPの提供状況は[README](../../README.ja.md)に記録する。統合稿の恒久アーカイブDOIは未割当である。

## 1. 主要結果の入口

|本文の対象|結果報告|原保存物|別実装での再計算記録|
|---|---|---|---|
|Ouroの未使用四文章・表3/4/7/8・図2/4/5|[AC08](../../reports/ac08-results-v0.1-2026-10-02.md)|[run](../../runs/loop-kv-ac08-20261002-095022-41c2554c/)|[audit](../../data/validation/ac08/)|
|Nanbeigeの未使用四文章・表3/4/7/8・図2/4/5|[NB05](../../reports/nb05-results-v0.1-2026-10-03.md)|[run](../../runs/loop-kv-nb05-20261003-090604-18baabf5/)|[audit](../../data/validation/nb05/)|
|同算術の橋渡し対照・表5・図1/3|[VF01](../../reports/vf01-results-v0.1-2026-10-02.md)|[run](../../runs/loop-kv-vf01-20261002-072452-1d1c10c0/)|[audit](../../data/validation/vf01/)|
|入力検査の集約・表6|[IV01](../../reports/iv01-results-v0.1-2026-10-01.md)|[run](../../runs/loop-kv-iv01-20261001-142907-0c002415/)|[audit](../../data/validation/iv01/)|

原runの `summary.json`、`run_status.json`、manifest／SHA256一覧は結果の入口。各auditには返却時の取り込み、配布前コードとの照合、再計算、正式値との対応がある。返却ZIP内のコードを無条件に実行して確認したものではない。

## 2. 主張・表・データ項目

|本文|機械可読の出所|採用した項目・単位|
|---|---|---|
|表4・図2左：Ouro memory|[AC08記述集計](../../reports/ac08-descriptive-statistics.json)|`memory` 内の `terminal_storage`、`allocated_peak`、`reserved_peak`、`recorded_prefill_inclusive_peak`。byteをMiBへ換算|
|表4・図2右：Nanbeige memory|[NB05記述集計](../../data/validation/nb05/supplementary-analysis.json)|各 `books[].terminal_kv_bytes` と `memory`。全文章・反復で同値|
|表5・図3：橋渡し対照|[VF01分析](../../data/validation/vf01/analysis.json)|`main_time_ms`、`timing.cuda_ms.*.main`、`main_memory_bytes`。主6組、独立文章は2|
|表7・図4：文章別品質|[本文用データ](paper-data-v0.3-2026-10-08.json)|`quality`。Ouroは255予測と終盤64、Nanbeigeは圧縮後8。PPLの直接平均はしない|
|表8・図5左：Ouro時間比|[AC08記述集計](../../reports/ac08-descriptive-statistics.json)|`time_ratio_summary` と `window_time_ratio_summary`。8対応比の中央値|
|表8・図5右：Nanbeige時間比|[NB05記述集計](../../data/validation/nb05/supplementary-analysis.json)|`timing.cuda_ms.decode`／`phase_sum`。文章内3対応比の中央値→4文章の中央値|
|表6：検査費用|[IV01記述集計](../../reports/iv01-descriptive-statistics.json)|18実入力の代表値と対応比。64fixtureの192判定を独立文章数にしない|
|式・実packed量|[NSN算術](../../scripts/loop_kv_nb05_primitives.py)、[codec](../../scripts/loop_kv_nb05_codec.py)|64×128区画。符号3072＋補助情報640＋header12＝3724 byte。両K/V、全head・slot、BF16末尾を合計|

図1は読み出し経路の模式図。図2〜5は本文用抽出データから作成した。図2の分母は各測定項目の非圧縮値、図3は各経路の水準値、図4は文章別PPL変化、図5は取得前に固定した単位で集計する対応時間比である。画像の元となるSVGも `figures/v0.2/` に保存した。本文用抽出・作図コードは[build_paper_assets_v0_2.py](code/build_paper_assets_v0_2.py)。読み込み元8ファイルのhashを本文用JSONに保存する。この図表生成は、前回の検証済み数値を本文へ対応づける作業であり、原logitsの再検証とは別である。v0.4で行ったNanbeige保存logitsのCPU照合は§8へ分けた。

## 3. 取得前計画と再計算の入口

- **AC08**：[設計](../../plans/loop-kv-ac08-design-v0.1-2026-10-02.md)／[取得前カード](../../plans/prespecified/loop-kv-ac08-v0.1-2026-10-02.md)／[検証器](../../scripts/verify_loop_kv_ac08.py)。
- **NB05**：[設計](../../plans/public/loop-kv-nb05-design-v0.1-2026-10-03.md)／[取得前カード](../../plans/public/loop-kv-nb05-v0.1-2026-10-03.md)／[検証器](../../scripts/verify_loop_kv_nb05.py)。
- **VF01**：[設計](../../plans/public/loop-kv-vf01-design-v0.1-2026-10-02.md)／[取得前カード](../../plans/public/loop-kv-vf01-v0.1-2026-10-02.md)／[検証器](../../scripts/verify_loop_kv_vf01.py)。
- **IV01**：[設計](../../plans/loop-kv-iv01-design-v0.1-2026-10-01.md)／[取得前カード](../../plans/prespecified/loop-kv-iv01-v0.1-2026-10-01.md)／[検証器](../../scripts/verify_loop_kv_iv01.py)。

数値の閾値は各検証器がimportする契約・数値モジュールと入力JSONに保持され、配布時hashと照合されている。AC08の公式逐次経路への一致、VF01の橋渡し一致、NB05のGQA数値予算は別の条件である。旧BF16 full／incremental一致検査の未達を、これらの成立で上書きしない。

再集計を行う場合は原runを保持し、検証器の副作用を確認して作業コピー上で実行する。とくにAC08検証器のCLIは `completion_verification.json` を書くため、原証拠へ直接実行しない。本文用の表と図だけを再作成する場合は、研究ルートから次を実行できる。

```bash
python3 papers/loop-kv/code/build_paper_assets_v0_2.py --research-root .
```

このコマンドは `papers/loop-kv/` のv0.2派生JSONと `figures/v0.2/` のPNG・SVGを再生成する。v0.1の稿・データ・図・コードは保持する。GPU、モデル、ネットワークは使わない。必要packageはmatplotlibとPython標準library。

## 4. 数値照合の区別

- **原値の保持**：原summary、logits、cache、予定表、停止記録は変更していない。
- **NB05派生float**：元の再計算では22項目に最大3.553e−15の差があり、取得前に固定されたrtol／atol 1e−12内。離散値・原配列の完全一致とは分ける。
- **VF01**：H2（数値）とH3（時間）は成立。H1は非連続fixture作成不備で未達のまま。[NC01](../../reports/nc01-results-v0.1-2026-10-02.md)は後続の別確認。
- **AC08**：実Qの橋渡し／融合は一致したが、既存復元器との差は残った。全系列logitsの方式間bit一致を品質成立条件へ置き換えない。
- **NB05**：捕捉16点の工程別誤差予算内という結果は、非圧縮attentionとの全bit一致を意味しない。

## 5. 前段と失敗記録の索引

以下は主要結果を形成した条件の説明と、追跡のための資料。本文の主標本へ合算しない。

|報告|内容|
|---|---|
|[P0-DIAG01](../../reports/p0-diag01-results-v0.1-2026-09-30.md)|P0小診断：10条件完了、BF16の合格候補は得られず|
|[P0-FP32-01](../../reports/p0-fp32-01-results-v0.1-2026-09-30.md)|FP32参照：4冊・全4周回の16条件が基準内|
|[P0R-P1A](../../reports/p0r-p1a-results-v0.1-2026-09-30.md)|P0R–P1A：BF16の経路検査が成立し、4周での量子化比較64条件を完了|
|[P0RB-P1B](../../reports/p0rb-p1b-results-v0.1-2026-09-30.md)|P0RB–P1B結果：1・2・8周で品質保持の5%目安内|
|[KV01](../../reports/kv01-results-v0.1-2026-09-30.md)|KV01結果：K側のPPL損失がやや大きく、両側の平均損失はほぼ加算的|
|[LP01](../../reports/lp01-results-v0.1-2026-09-30.md)|LP01結果：同量の圧縮を置く周回による差は、固定した平均幅内だった|
|[LC01](../../reports/lc01-results-v0.1-2026-10-01.md)|LC01：継続圧縮は2,048 tokenでも品質目安内、区間差は一様ではない|
|[LC02](../../reports/lc02-results-v0.1-2026-10-01.md)|LC02：未使用の8冊でも、継続圧縮の品質目安を満たした|
|[PK01](../../reports/pk01-results-v0.1-2026-10-01.md)|PK01結果：実3bit保存の容量・予測・速度と、数値照合の未達|
|[PD01](../../reports/pd01-results-v0.1-2026-10-01.md)|PD01結果：単独部品では3bit展開が長く、検査待ちの集約差は約5%|
|[PU01](../../reports/pu01-results-v0.1-2026-10-01.md)|PU01結果：3bit展開の集約で、値と容量を保ちながらモデル経路を約18.5%短縮|
|[PU02](../../reports/pu02-results-v0.1-2026-10-01.md)|PU02結果：圧縮更新を3回通しても、値一致と約19%の時間短縮を維持|
|[RF01](../../reports/rf01-results-v0.1-2026-10-01.md)|RF01結果：復元値と検査を保ち、前回方式からさらに約15.7%の時間短縮|
|[QK01](../../reports/qk01-results-v0.1-2026-10-01.md)|QK01結果：K復元と点数計算の融合は正常完走、部品時間を約32.7%短縮|
|[AC01](../../reports/ac01-results-v0.1-2026-10-01.md)|AC01：全測定後のsoftmax照合で停止。型エラーではない|
|[AD01](../../reports/ad01-results-v0.1-2026-10-01.md)|AD01結果：FP32の微差が再現し、準備後の融合attentionは非融合比20.2%短縮|
|[AC02](../../reports/ac02-results-v0.1-2026-10-01.md)|AC02結果：全長さの準備後、モデル時間が非融合比で約7.6〜7.7%短縮|
|[AF01](../../reports/af01-results-v0.1-2026-10-01.md)|AF01結果：softmax・V合成まで融合し、部品時間を26.8%短縮|
|[AC03](../../reports/ac03-results-v0.1-2026-10-01.md)|AC03結果：融合によるモデル時間7.8%短縮と、非量子化に対する交換条件|
|[DC01](../../reports/dc01-results-v0.1-2026-10-01.md)|DC01結果：V復元と入力検査の費用が見えた|
|[AC04](../../reports/ac04-results-v0.1-2026-10-02.md)|AC04結果：入力検査の集約でモデルdecode時間15.4%短縮|
|[AC05](../../reports/ac05-results-v0.1-2026-10-02.md)|AC05結果：3回の圧縮更新後も、出力を保って検査集約の短縮が残った|
|[AC06](../../reports/ac06-results-v0.1-2026-10-02.md)|AC06結果：未使用4冊でも、品質と検査集約の時間短縮を確認|
|[NC01](../../reports/nc01-results-v0.1-2026-10-02.md)|NC01結果：正常入力は通り、非連続入力は計算前に止まった|
|[AC07](../../reports/ac07-results-v0.1-2026-10-02.md)|AC07結果：V融合による短縮がモデル全体でも残った|
|[NB01](../../reports/nb01-results-v0.1-2026-10-02.md)|NB01結果 — Nanbeigeの小接続確認が成立|
|[NB02](../../reports/nb02-results-v0.1-2026-10-02.md)|NB02結果 — Nanbeigeの実3bit KV保存と全量復元が成立|
|[NB03](../../reports/nb03-results-v0.1-2026-10-02.md)|NB03結果 — Nanbeige GQAへの融合部品の移植が成立|
|[NB04](../../reports/nb04-results-v0.1-2026-10-03.md)|NB04 結果 v0.1 — 2026-10-03|

停止と修正の対応は上記の公開報告を参照。内部の運用台帳は公開対象に含めない。本文はこの履歴を実験の独立標本数として積み上げない。

## 6. 今回稿と統合稿の境界

本稿は再帰型モデルでの実装・評価を詳述し、[NSN統合稿 v1.0](../nsn-integrated/NSN-2026-10-29-JA.md)は構成・適用条件・実保存量・利用費用をまとめる。既公開NSNの版照合は[版対応記録](../nsn-integrated/publication-provenance-v0.1-2026-10-03.md)に保存済み。著者作成の論文・図・データはCC BY 4.0、コードはApache-2.0とし、第三者条件を保持する（[利用条件](../../LICENSES.md)）。リポジトリは[https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn)。公開範囲と別配布ZIPの提供状況は[README](../../README.ja.md)に記録する。統合稿の恒久アーカイブDOIは未割当である。

## 7. v0.3の短い実装と文献の照合

本文コード1の[nsn_int3_core.py](code/nsn_int3_core.py)は、importと関数定義を除く関数本体が10行。今回の[凍結済み算術](../../scripts/loop_kv_nb05_primitives.py)の `nsn_encode`／`nsn_decode`／`nsn_qdq` を参照した。方向の分母 `1e-12` とFP16補助情報の丸めを含む今回の規則に合わせており、旧NSNの説明コード全体をそのまま転載したものではない。

[照合結果](../../data/validation/qdq-core/core-validation.json)は合成8種×3 dtype（FP32／FP16／BF16）の24条件で、CPU上の参照出力と全bitが一致した。全ゼロ、定数、ゼロ行、ノルムの分散、stride付き入力等を含む。モデルの品質測定、任意入力への完全性証明、GPU融合kernelとの全bit一致の試験には数えない。参照sourceと短いコードのSHA256、fixtureごとの結果を保存した。[照合スクリプト](../../data/validation/qdq-core/check_core.py)は参照コードから三関数だけを読み出してCPUで実行する。

実保存には[codec](../../scripts/loop_kv_nb05_codec.py)とcache・検査・attention経路が別途必要となる。コード1のuint8配列を実3bit保存と数えない。§4〜6の数値、図5点、表4〜8、[本文用JSON](paper-data-v0.3-2026-10-08.json)はv0.2から変更せず、同じ派生資産を参照する。

追加した一次資料と採用範囲は[関連研究照合v0.2](related-work-review-v0.2-2026-10-03.md)。重み・活性値、周回方向の低次元化、周回間残差の量子化を区別する。新しい文献の採用によって、自分の測定条件や先行runの成否を変更していない。


## 8. 数値条件と再現入口の区別（v0.4）

本文付録Bの表B1は取得前の固定コード、表B2は保存済み診断値に対応する。[数値照合JSON](numerical-audit-v0.4-2026-10-03.json)に最大差・通過点数・出所hashを保存した。誤差尺度は同じ圧縮入力に対する復元・計算経路の整合性を測る。非圧縮からの品質損失は本文表7のPPLで評価する。

|判定|固定コードの入口|参照の区別|
|---|---|---|
|復元K/V、QK score|[QK01契約](../../scripts/loop_kv_qk01_contract.py) `k_metrics`・`score_metrics`／[NB05数値](../../scripts/loop_kv_nb05_numeric.py) 同名関数|Ouroのscore参照は凍結復元K、Nanbeige融合のscore参照は融合で復元したK。Nanbeigeは復元K/Vの誤差を別に検査|
|context|[AF01契約](../../scripts/loop_kv_af01_contract.py) `context_metrics`／NB05数値の同名関数|保存FP32 scoreのFP64 softmaxと、その経路の復元V|
|参照Vとの差を含むcontext|[VF01契約](../../scripts/loop_kv_vf01_contract.py) `context_budget`／[VF01検証](../../scripts/verify_loop_kv_vf01.py) `numerics`|V復元差を加えた予算。QK誤差はscore検査で扱う|
|SDPA出力|NB05数値の `sdpa_metrics`|各SDPA経路自身の復元K/VによるFP64 attention。尺度は重み付き絶対値和|
|橋渡し、通常呼出し、実forward|[AC08診断](../../scripts/loop_kv_ac08_audit.py)／[NB05診断](../../scripts/loop_kv_nb05_diagnostic.py)|同じ入力・算術とのbit一致。Nanbeigeは橋渡しK/Vの一致と、最終出力の尺度付き誤差を分ける|
|NLL|[AC08検証器](../../scripts/verify_loop_kv_ac08.py)／[NB05検証器](../../scripts/verify_loop_kv_nb05.py)|保存logitsによる再計算。本文の品質評価の独立文章数とは別の確認件数|

今回の改訂では、AC08・NB05・VF01の保存診断集計を抽出し、NB05の保存FP32 logits 48系列（432予測）をCPUで再計算した。最大NLL差は7.106e−15で既存の相対／絶対許容値1e−12内。[予測ごとの照合](../../data/validation/nb05/nb05-nll-checks.json)を保持する。AC08の8,925予測の最大差は、以前の独立検証の保存結果から引用した。今回GPU再実行やAC08 raw配列の全再検証をしたとは扱わない。

読み手の目的に応じ、次の三つの入口を分ける。

1. **本文の主要数値・図の再生成**：§2・3の本文用JSONと作図コード。既存集計を読む。
2. **診断値の抽出と保存NLLの照合**：[CPUスクリプト](code/extract_numerical_audit_v0_4.py)。下記は別出力先へ書き、原run・正式集計を更新しない。必要packageはNumPyとPython標準library。出力は補足JSONと予測ごとの照合JSONの2ファイル。契約コードのhashも補足JSONへ含める。
3. **元実験の数値検証・GPU再実行**：§3の検証器、原runのraw配列、取得前計画、配布コード。CPU抽出だけではkernelの再実行にならない。GPU環境を必要とする作業はこの改訂では行っていない。

```bash
python3 papers/loop-kv/code/extract_numerical_audit_v0_4.py \
  --research-root . --output-dir /tmp/loop-kv-numerical-audit
```

このスクリプトは、保存summaryから工程別最大差を読み、NB05 rawファイルの記録hashを確認してNLLを再計算する。実験用Pythonコードはimport／実行しない。出力先に同名ファイルがある場合は停止する。

## 9. 主な実行経路と検査・同期位置

表8は以下の**検査付き経路一式**を比較する。同じ名称の「融合」でも変更範囲が異なる。計時外に検査を移して得た短縮ではない。

|モデル・経路|復元算術とattention|入力・出力検査とホスト確認の位置|
|---|---|---|
|Ouro `NATIVE`|BF16 K/V。公式SDPA呼出し|decode位置と全許可maskを確認。Q/K・Vの構造／有限性、attention出力の有限性を確認|
|Ouro `PACKED_GROUPED`|Kをattention内で復元。Vは凍結RF01復元器で全長配列にする。専用QK・softmax・V合成|V全量復元後の有限性確認。Q・K補助情報／tail・復元Vを集約検査し、flagをホストで確認。融合計算のflagと最終出力を確認|
|Ouro `PACKED_KV_FUSED`|K/Vを区画単位で内部復元。V復元には融合用算術を使う|Q・K/Vのpacked構造、補助情報／tailの有限性を検査。計算flagと出力を確認。外側のモデル接続部でも出力有限性を確認|
|Nanbeige `NATIVE`|BF16 K/VをGQA用に6倍対応づけしてSDPAへ渡す|Q・K/Vの共通構造／有限性検査、集約flagをホスト確認、出力有限性を確認|
|Nanbeige `PACKED_SDPA`|PyTorchの符号展開と参照NSN復元でK/V全長配列を作り、GQA対応づけ後SDPAへ渡す|NATIVEと共通のraw入力検査・flag確認。復元／出力を検査|
|Nanbeige `GQA_FUSED`|Q headをKV headへ6対1に対応づけ、K/Vを区画内復元してattentionを計算|共通raw入力検査後、partial／merge kernelのflagと出力を確認|

Ouroの経路は[モデル接続](../../scripts/loop_kv_ac08_attention.py)、[入力集約検査](../../scripts/loop_kv_iv01_kernel.py)、[V融合](../../scripts/loop_kv_vf01_kernel.py)、[V入力検査](../../scripts/loop_kv_vf01_checks.py)、[RF01復元](../../scripts/loop_kv_rf01_kernel.py)に対応する。Nanbeigeは[モデル接続](../../scripts/loop_kv_nb05_attention.py)、[kernel経路](../../scripts/loop_kv_nb05_kernel.py)、[共通検査](../../scripts/loop_kv_nb05_checks.py)に対応する。

Nanbeigeの `TILE_SDPA` は、融合と同じ算術で全長K/Vを作ってSDPAへ渡す**局所診断の橋渡し**であり、本文表8の主モデル比較の一群ではない。Ouroの橋渡しもV復元算術をそろえた部品比較として表5に分けた。

`bool(tensor)`、位置の `item()`、有限性flagの確認にはGPUの結果をCPUへ返す待ちが含まれる。CUDA eventとwall時間の区切りは、[AC08実行器](../../scripts/loop_kv_ac08.py)と[Nanbeige計時器](../../scripts/loop_kv_nb05_measure.py)で管理し、計測開始前と終了時に同期する。証拠のCPUコピー・ファイル保存は当該時間の外に置く。経路ごとに復元算術・検査対象・呼出し数が異なるため、表8からV全量配列の除去だけの因果効果を取り出さない。

SDPAは**自動選択、個別backend未同定**。Ouroの[実行設定](../../runs/loop-kv-ac08-20261002-095022-41c2554c/execution_settings.json)はflash／efficient／math／cuDNNの有効化状態を記録するが、各呼出しで選ばれたbackendではない。Nanbeigeの[環境記録例](../../runs/loop-kv-nb05-20261003-090604-18baabf5/books/b0/environment.json)もPyTorch default方針までを記録する。保存された診断から選択backendを推定で補わない。

## 10. 計時位置と圧縮更新の順序

ここで (L) はforwardで処理済みのtoken数、すなわちattention時のKV長。token配列の添字は0から数える。

|Ouroの集合|KV長 (L)|個数|
|---|---|---:|
|decode全体|769〜1023|255|
|最初の除外位置|769〜776|8|
|捕捉による除外位置|769, 831, 833, 895, 897, 959, 961, 1023|8|
|二つの除外集合の共通要素|769|1|
|主計時|decode全体から二つの除外集合の和集合を引く|240|
|forward後の追加圧縮|832, 896, 960|3|

[固定入力JSON](../../scripts/loop_kv_ac08_inputs.json)の `timed_lengths`／`capture_lengths`／`flush_post_lengths`と一致する。圧縮が走る832・896・960は主計時に含まれる。全255予測を品質に使い、240位置を主時間に使う。捕捉は各位置のslot 0・95・191。除外位置を変えて数値を作り直してはいない。

圧縮更新は[Ouroの基底cache](../../scripts/loop_kv_pk01_cache.py)と[Nanbeige cache](../../scripts/loop_kv_nb05_packed.py)の `post_prefill`／`post_decode` に従う。処理順の要約は次のとおり。

```text
prefill: 非圧縮KVでattention → モデル出力
post_prefill: 末尾64 token以上を残し、古い完全64-token区画を圧縮
各decode:
    新K/VをBF16末尾へ追記
    packed prefixとBF16末尾を使ってattention → モデル出力
    post_decode: 末尾が128 token以上なら、古い64 tokenを圧縮
    新しい末尾は残りのBF16値として保持
```

Ouroはprefill 768 tokenの後に704 tokenを圧縮する。L=832のattention時には圧縮704＋末尾128、そのforward後に圧縮768＋末尾64となる。同様にL=896・960で64 tokenずつ追加し、L=1023の終端は圧縮896＋末尾127。FORWARDとPOSTを別計時し、主時間では両者を足す。prefillと初期圧縮は主時間から分け、初期費用込みの指標へ含める。

Nanbeigeはprefill 128 token後に64 tokenを圧縮し、8回のdecode後は圧縮64＋末尾72。今回の系列では追加圧縮の条件へ達しない。decodeだけの短縮と、初期圧縮を含む時間の増加が両立するのは、この短い利用区間での結果である。

## 11. 材料・tokenizer・予測対象

PG19の候補一覧はrevision `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b` に結び付けた。実テキストは `https://storage.googleapis.com/deepmind-gutenberg/train/{book_id}.txt` から取得した。固定候補・seed・除外条件は§3の取得前計画と各 `selection_lock.json`、実際の原URL・本文hash・token列・token hashは以下の `materials.json` を参照する。

|モデル|4文章のbook ID|tokenizer固定revision|入力・評価対象|
|---|---|---|---|
|Ouro|38233, 57533, 3921, 58192|`fcfcfbc429c7b10ee73dfc718033351d8936d2a8`|先頭1024 token。prefill入力の添字0〜767、主予測の正解token添字769〜1023（255個）。prefillの予測先768は主品質から除く|
|Nanbeige|9667, 26438, 45277, 28881|`4a38e817c14b9f2c4b69b1e05cbda90395872f6b`|先頭137 token。prefill入力の添字0〜127、主予測の正解token添字129〜136（8個）。prefillの予測先128は主品質から除く|

- Ouro：[materials](../../runs/loop-kv-ac08-20261002-095022-41c2554c/materials.json)／[selection lock](../../runs/loop-kv-ac08-20261002-095022-41c2554c/selection_lock.json)／[選定・token化コード](../../scripts/loop_kv_ac08_materials.py)。
- Nanbeige：[materials](../../runs/loop-kv-nb05-20261003-090604-18baabf5/materials.json)／[selection lock](../../runs/loop-kv-nb05-20261003-090604-18baabf5/selection_lock.json)／[選定・token化コード](../../scripts/loop_kv_nb05_materials.py)。

取得bytesをUTF-8でdecodeし、`add_special_tokens=False, truncation=True`、`max_length=1024`または137でtoken化した。取得後に冒頭・前書きを除去したり、空白を独自に整形したり、chat templateやBOS／EOSを手作業で加えたりしていない。tokenizer内の正規化は固定版の動作に従う。Ouroは固定revisionの `AutoTokenizer` の既定設定、Nanbeigeは `use_fast=False, legacy=True` で読み込んだ。今回から前処理を変更したものではない。

主品質のtoken列とraw logitsの保存対象は同じ件数ではない。例えばNanbeigeの保存logitsは、prefillの1予測を含む9予測×48系列をCPU照合した。主品質の8予測と、数値整合性の9予測を混ぜない。

## 12. 継続評価の証拠と追補の再生成

本文の継続評価はNB06として取得前に固定した。[設計](../../plans/loop-kv-nb06-design-v0.1-2026-10-08.md)、[取得前カード](../../plans/prespecified/loop-kv-nb06-v0.1-2026-10-08.md)、[入力契約](../../plans/loop-kv-nb06-input-contract-v0.1-2026-10-08.json)、[実装](../../scripts/loop_kv_nb06.py)、[固定検証器](../../scripts/verify_loop_kv_nb06.py)、[原結果](../../runs/loop-kv-nb06-20261008-075410-5621380a/summary.json)、[入力token](../../runs/loop-kv-nb06-20261008-075410-5621380a/materials.json)に対応する。

[結果再集計](../../reports/nb06-results-v0.1-2026-10-08.json)のquality・timing・mean_additive_cost_cuda_msが、本文の品質表、累積時間表、記述的費用内訳の入口である。[再集計コード](../../scripts/analyze_loop_kv_nb06_results.py)は原stepから値を照合する。32実Q点の診断は原summaryのbooks[].actual_q、メモリはbooks[].memory_peak_bytesとterminal_kv_bytesから確認できる。[照合記録](../../data/validation/nb06/verification-summary.json)と[本文用NB06数値表](numerical-audit-nb06-v0.1-2026-10-08.json)も保持する。

初回の表と図1–5は従来の測定を保持し、追補は別節・図6・別版JSONへ加える。[追補生成器](../../scripts/build_nb06_paper_supplements.py)と[固定差分原稿](../../scripts/nb06_paper_supplements_v0_1.json)は、基準本文・保存集計・原summary・図6のhashを照合し、日英本文、図、本文用JSONを新しい出力先へ再生成する。研究ルートで次を実行する。

~~~bash
python3 scripts/build_nb06_paper_supplements.py \
  --research-root . --output-dir /tmp/nb06-paper-regeneration
~~~

これは本文と表・図の再生成である。生logits・数値検査の再実行には原runの作業コピーと上記固定検証器を使う。取得元は同じ4文章で、旧NB05との品質・時間のプールは行わない。原ZIP SHA256は2ef013414a6a7130bb55ddd0ff32328698b965a1f3347fa023b2400d5d2246fa、原summary SHA256は6da2ac29e8490ef5a36364b5637ce56d47bc00554605674a38f3ec93bd4ddc8a。
