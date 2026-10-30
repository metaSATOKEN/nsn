# 抽出仕様と原記述の採否

2026-10-03／v0.1。[報告](extraction-report-v0.1-2026-10-03.md)と[再集計表](evidence-tables-v0.1-2026-10-03.md)の方法・証拠への入口。ここで定めるのは統合のための抽出規則であり、原実験の取得前仕様を変更するものではない。

## 1. 対象と元記録

|証拠ID|今回再集計した対象|確認の粒度|
|---|---|---|
|E1|modeq Phase 14、16〜19|主表8モデルと四方式32セル。保存PPL・差・丸めを照合。prior_*と後続要約の重複は不加算|
|E2|Phase 20、21、23、28b、28d|dtype、区間、サイズ、bit幅の境界47セル。8モデルの保存CV平均・相関|
|E3|SADP E5 holdout、holdout causal faithful、W1d、W1e|品質88セル・窓値2,640件・対応差54組・4未実行構成。元手順の区間を再現|
|E4|Phase 15、26、27|保存量10行、packed PPL2モデル、時間10行。計時範囲をコードから読む|
|E5|K1、K1b、K1c|480セル（475有効・5 OOM）。保存済み中央値の形状別対応比。100反復のraw計時は未保存|
|E6|AC08、NB05、VF01、本文用データ・既存数値監査|下記§4の再集計と既存hash照合。IV01の工程検査数値は既存稿・監査を継承し、新規全fixture再実行はしていない|
|E7|MMLU、needle|所在の一覧のみ。今回は再採点せず、付録候補のまま|

参照元は研究内の[modeq](../../modeq/)、[SADP](../../Substrate-Aware-Diagnostic-and-Prescription/)、[今回研究](../loop-kv/README.md)。372参照ファイルのSHA256は[source-manifest.json](data/extraction-v0.1/source-manifest.json)。今回の原値・コード・下位記録を対象とし、既存数値監査が参照する全raw logitsのhashは元の監査にも残る。元ファイルを修正しない。

機械表の`origin.source`は研究ルートからの相対パス、`pointer`はJSON Pointer、`sha256`は参照元内容のhash。JSONLの場合は空pointerでファイル全体を指す。NB05の子集計は`source_directory`内のstep-000〜008.jsonと対応する。モデル形状をまとめたkernel_summary等は、同じrunのkernel_cellsからの派生集計である。

## 2. 実装・測定の違い

|項目|旧NSNの主品質|SADPの採用品質|旧NSN実packing／時間|SADP K1c|今回の実モデル|
|---|---|---|---|---|---|
|入力|WikiText-2 test|WikiText-2 testの固定seedの窓|品質はWikiText-2、packing量は合成tensor|合成K、モデル形状|PG19 training split、研究内で未使用の固定文章|
|主単位|最大10窓、1024長、stride512|30窓、1024〜65536長|PPL集計／3技術反復|形状・方式ごとの計時中央値|各4文章、Ouro2／Nanbeige3技術反復|
|採点|512 prefixを圧縮し、後半512予測を採点|hookを通した各窓の相対PPL|Phase 26の2モデル、Phase 27の50逐次step|モデルPPLは測らない|Ouro各255／Nanbeige各8予測|
|K/V|両方へ同じrecipe。Kのみ・Vのみの追試は別セル|NSNは両方NSN、plainはKチャネル別・V token別|同じ符号でも補助情報のFP16保存で模擬と差|Kのみを復元してQK score|K/Vを別保存、周回・層を保持|
|位置|prefill cache内のK（RoPE後）|k_projまたはk_norm後、RoPE前。Vはv_proj|cache|RoPE操作を含まない合成部品|各モデルの公式attention接続点|
|区画|prefix全体|A0全窓、A1 128区画、A2末尾128を高精度、可能な場合A3 post-norm|旧cacheのtoken軸|形状に応じた全長統計|64 token区画＋高精度tail|
|保存形式|模擬QDQ|模擬QDQ|実3bit＋FP16補助情報|1 code＝uint8、実4/5bitstreamではない|実3bit＋FP16補助情報＋header|
|終点|PPL|PPL|PPLまたは逐次モデル呼出し|QK score|softmaxとV利用を含むattention、後続予測|

旧主品質コードは、非空行を改行で結合した先頭48,000文字をtoken化し、先頭512をprefillする。後半512の最初の予測には圧縮前の最終logitsが使われるため、512予測すべてが圧縮後のattentionを通ったわけではない。sourceの丸めを超える精度を推定しない。[四方式コード](../../modeq/experiments/phase18_ablation_vs_industry_naive.py)。

SADPの`kivi`というキーはローカルplain-asymmetric比較実装を指す。公式KIVIのpacking・kernel・全設定を実行したという証拠にはしない。A0は窓全体からmin/maxを得る。A1〜A3のスケール範囲を変更した評価も、online serving全体と同一とは扱わない。[holdoutコード](../../Substrate-Aware-Diagnostic-and-Prescription/experiments/phase_holdout_causal_faithful.py)、[W1dコード](../../Substrate-Aware-Diagnostic-and-Prescription/experiments/phase_w1d_causal_faithful_longctx.py)。

共通原理は同じでも、中間dtype、保存丸め、復元方向の正規化の実装は一致していない。旧QDQの構成式、FP16補助情報の実保存、effective normで計算するscore部品、今回の橋渡し対照を段階別に説明する。

## 3. 品質・区間の算出規則

- 旧NSN：保存PPLから`100 × (PPL_quant / PPL_base − 1)`。主表C/Dは初出の保存ΔPPLを使う。四方式表は保存PPLの差から再計算し、丸められた後続表との約1e−4の差を区別する。原判定は＋5%未満。独立した窓値がない表に区間を新造しない。
- SADP：窓別の相対PPL変化を平均。窓値の平均と標本標準偏差は別のPython標準ライブラリ実装でも確認。区間は原コードの`bootstrap_ci`関数だけをASTで取り出し、CPU上で5,000再標本化を再現した。元実験モジュール全体はimport・実行しない。
- SADPのseed：各品質セルは元seed 20260510＋1、E5の対応差は＋2、他の対応差は＋7。元の両側95%百分位区間を保持する。K/V方式群の差は同じ窓のplain−NSN。E5のthree-tier mask差は別の向きであり、機械表の`direction`に従う。
- 非有限値：NaNを含むセルは削除せず残す。JSONには非標準NaNを出さず文字列`"nan"`とし、`finite_windows`、`invalid_windows`を併記する。OLMoE A3 plainの30/30非有限は有効な0悪化として扱わない。
- 窓の依存：全15窓配置で開始位置と長さを照合し、重複する窓対を数えた。原bootstrapの窓単位を再現する作業であり、依存を補正した新しい推定ではない。モデル・窓・token・技術反復を足して総合Nを作らない。
- 今回二モデル：同じ文章の反復をNLLの独立な観測へ加えず、文章ごとのNLL差を等重みで平均して`100 × expm1(mean ΔNLL)`。各4文章という範囲を維持する。

5%目安とSADPの0.5%判定、実装の数値許容値は用途が違う。SADP機械表には、点が±0.5内、区間全体が±0.5内、区間上限が＋0.5未満を別booleanで保存した。この説明用の再判定を原試験の成功基準へ書き戻さない。

## 4. 保存量・速度と検証の深さ

旧packingの3.140625 bit/要素は、[Phase 15コード](../../modeq/experiments/phase15_real_compression_bench.py)の合成単一sliceから算出。INT4行は理論値であり実packing実測ではない。Phase 27は保存3反復の中央値から非圧縮比を再計算した。compiledは通常5stepに対し10stepをwarmupし、初期packingは計時外。異なる開始長を含む実装経路の比較である。

SADP K1/K1b/K1cは保存済みのセル中央値から形状を対応させ、FP16/NSN5比の中央値を取る。K1cの相対誤差は同じ量子化値をFP32復元した参照との実装整合で、非圧縮Kへの量子化誤差とは違う。最大保存誤差は約5.583e−7、固定閾値0.01以内。raw100反復から新しいCIは計算していない。

VF01は原540計時から、①各試料・query・armの5反復中央値、②同じqueryの方式比、③非zero 3queryの中央値、④6主試料の中央値の順。2文章から3slotずつの試料で、6独立文章ではない。旧H1未成立を維持する。今回の抽出器の初稿ではqueryを先に潰した比との不一致を検出し、原契約どおりの順へ修正した。元の結果値は正しく、変更していない。

AC08は保存8比較ブロックの反復NLL一致と文章PPL・対応時間比を照合。NB05は主4文章×3反復×3方式×9 forward＝324 step JSONから36系列を再集計した。NB05のgate等を含む全60系列・540 forwardを今回すべて再解析した、という意味ではない。

今回稿の[数値監査](../loop-kv/numerical-audit-v0.4-2026-10-03.json)に記録されたsource・契約・raw logitsのhashを照合した。raw logitsのNLL再計算は同一プロジェクト内で以前に行った検証を継承する。旧系列についてraw logitsの独立再推論やGPU計時の再実行はしていない。

## 5. 原記述の採否

|項目|裁定|統合稿での表現|
|---|---|---|
|A1 模擬QDQと実保存|有限差を採用|整数符号の可逆性、復元値の差、PPL差を別に報告。Phase 26の`abs`が負値の行は符号付き差と真の絶対値を別列にする|
|A2 K名称とV介入|方式群として採用|K/V両方が変わる。Kだけの因果効果へしない|
|A3 ±0.5%のall表現|一括表現を不採用|E5 Qwen plain INT5＋0.621%とCIをセル別表示。長区間コードの上限判定とも分ける|
|A4 名目bitと実保存|保存形式を明記して採用|旧合成slice、SADP uint8 score、今回bitstream＋attentionを区別|
|A5 代表速度|全有効セルと計時範囲を採用|Qwen形状のほぼ同速、5 OOM、初期費用外、compile warmup差も残す|
|A6 公開状態|ローカル来歴のみ確定|旧コピーは重複計上しない。公開版・DOI・ライセンスは公開前に確認|
|A7 必要性・厳密ノルム・機構|広い断定は不採用|比較集合内の構成差、有限精度での復元、探索的相関として書く|
|追加：Gemma30窓|取得上限と実数を分離|20/22窓、10,240/11,264予測|
|追加：Aの成功モデル名|セル原値を採用|AはMistral・GPT-J・Qwen3.5の3モデル。Llamaの＋6.03%は5%を超える|
|追加：bitの奇偶性|旧要約の説明を不採用|Qwen2.5-7Bの2bit悪化は、4/6bitを含む比較から奇偶一般則を支持しない|
|追加：NaN fallback|変更履歴ごと採用|OLMoE A3 plainは未成立、2026-06-24の派生判定でA2を採用。取得前からA2だったとは書かない|
|追加：窓重複|依存を明記|既存の窓bootstrap区間を再現した値。独立文章数の主張や新しい補正済み推定をしない|

## 6. 来歴・環境の未確定項目

古いrunのモデル・tokenizerの完全なrevision、実行時GPU型番、依存パッケージをすべて再構成できるログは、今回の選択結果には揃っていない。旧主条件はrequested FP16、実dtype未記載のセルは`unknown`。Phase 20が記録したQwen3.5のBF16は、その後続診断の情報として併記する。

SADP再評価3ファイルはTransformers 5.12.1、E5はcompiler/preregistrationの短SHAを記録するが、HFモデルrevisionや全GPU条件の代替にはならない。これらの既存metadataは[protocol-specifications.json](data/extraction-v0.1/protocol-specifications.json)に分けて保持する。今回二モデルの固定revision・前処理・数値契約は[既存S1](../loop-kv/evidence-index-v0.4-2026-10-03.md)を採用する。SDPAの実選択backendは未同定。

この欠落のため旧実験の完全な同一環境再実行を保証する書き方は避ける。一方、保存値から本表を再生成する経路は固定できた。

## 7. 再生成

研究ルートで実行する。PythonとCPU版PyTorchが必要で、今回照合はPython 3.13.5／PyTorch 2.11.0。モデル・CUDA・HF認証は使用しない。出力先は原実験フォルダの外に指定する。

```bash
python3 papers/nsn-integrated/code/extract_existing_evidence.py \
  --research-root . --output-dir /tmp/nsn-integrated-check
python3 papers/nsn-integrated/code/build_extraction_tables.py \
  --data-dir /tmp/nsn-integrated-check \
  --output /tmp/nsn-integrated-tables.md
```

抽出器は原データの分母・元値・hash等を照合し、不一致なら出力を残して失敗終了する。今回の最終実行は1,732照合・不一致0。チェックの数は研究の標本数ではない。表生成コードは数値を機械表から取る。方法や採否の文章は上のコード・原記録照合に基づく編集文で、実験結果とは別に版管理する。

## 8. 次工程への裁定

R1〜R5（来歴、主要数値、実装差分、相違点、統計の定義）は、今回採用するE1〜E6の粒度で記述可能になった。全135結果の完全監査やraw logitsの全系列再計算を達成したという意味ではない。

R6の関連研究・外部一次資料の指定版確認、旧NSN公開版との対応、恒久パッケージと利用条件は継続課題。E7は補助候補で、採用するなら先に監査する。これらを作業欄に残し、C1〜C4の範囲に沿った日本語草稿へ進む。新実験は、機構や普遍的優越性など現在より強い主張を選ぶ場合に改めて設計する。
