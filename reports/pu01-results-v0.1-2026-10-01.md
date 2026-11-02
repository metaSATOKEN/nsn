# PU01結果：3bit展開の集約で、値と容量を保ちながらモデル経路を約18.5%短縮

2026-10-01 / v0.1。LOOP-KV-PU01。**固定範囲を完了し、正式状態は`COMPLETE_COMPARISON`。** 同じ3bit保存形式のまま、整数codeの展開を一つのGPU kernelにまとめた。展開単体の時間は約89%、検査込みの全復元は約42%、モデルのforward＋postは約18.5%短くなった。従来方式とのlogits・固定時点のKV・保存NLLの一致と、実KV storageの約70%削減を維持した。

## 1. 何を変えて比べたか

比較した三方式は、同じ量子化値を使う。FAKEは復元済みの値をBF16で持つ。PACKEDは3bit code等を持ち、従来の展開処理で読み出す。FUSEDは同じpacked bufferを持ち、3bitからuint8 codeへの展開だけを単一Triton kernelへ置き換える。

浮動小数点のNSN復元、BF16化、毎回その場の有限値検査、tailの連結、attention、CPU FP64採点は保った。BLOCK1024・4warps・autotuneなし。実環境はRTX PRO 6000 Blackwell Server Edition、torch 2.11.0+cu128、Triton 3.6.0。Tritonはtorch配布metadataの要求版と一致した。

Ouro-1.4Bの固定revision `fcfcfbc429c7b10ee73dfc718033351d8936d2a8`、BF16、8周。既知2冊3752／31131、prefill768、終端cache848、各80予測。3752はFAKE→PACKED→FUSED、31131はFAKE→FUSED→PACKEDの固定順で、それぞれ新しいcacheを使った。

固定設計 (`plans/loop-kv-pu01-design-v0.1-2026-10-01.md`)／取得前カード (`throughline/prereg/loop-kv-pu01-v0.1-2026-10-01.md`)。取得前の「展開単体は短くなる」という予想に沿った。モデル短縮は可能性として置き、倍率の予測はしていなかった。

## 2. 完了と値の照合

| 項目 | 結果 |
|---|---|
| 実行状態 | COMPLETE_COMPARISON、worker計時290.50秒（約4分51秒） |
| GPU接続 | 既存cache fixture8件、新kernel fixture17件、7軌道・455予測、5比較が合格 |
| 主比較 | 6/6軌道、480/480予測、欠測0・全有限 |
| 部品比較 | 12/12標本、216/216計測区間、4,320測定呼出し |
| 部品の前後ゲート | 24/24記録で全6経路のbyte一致・入力buffer不変 |
| 独立FP64採点 | 935/935件が絶対2e-5以内、最大差3.2731×10⁻¹⁴ |
| 原記録と配布物 | 78ファイル中77hash、配布11埋込ファイル、948forwardのraw logitsを照合 |

両冊のFAKE/PACKED/FUSEDは、各81forwardの全語彙logitsと保存NLLが完全一致した。KVは固定4時点（768／831／832／848）×全192slotのK/V digestが一致した。異常検査を減らして速くした結果ではない。

別の集計器は返却workerをimportせず、保存BF16のbit列からFP64で再採点し、時間・容量・部品集計を原記録から作り直した。新kernelの16入力fixtureについて、独立のbit列分解や固定乱数から正解codeのhashを再計算して一致を確認した。ゼロ受理・NaN拒否を含むGPU上の実行自体は保存ゲートの証拠による。手元でGPUを再実行したものではない。

部品測定に使った12標本の参照値＋tailは、実モデルPACKED終端の該当slotのdigestに接続できた。全復元の中間出力を手元で再生成した検証とは区別する。

再hashしたコピーに、NLL誤記、入力変更ゲートの不合格、部品集計値の変更、実cacheと標本の対応不一致、計測区間欠落、kernel正解hash不一致を入れた6負例は全拒否。原本は受理した。検算・検査記録 (`throughline/audits/loop-kv-pu01-results-v0.1-2026-10-01`)。

## 3. モデル経路まで時間短縮が届いた

初期decode8回を除く72回のforward＋post（flush込み）を集計した。CUDA Eventの平均、単位はms/token。モデルロード、初回コンパイル、CPU採点、保存、snapshot監査、部品測定はこの時間の外にある。

| 冊 | FAKE | 従来PACKED | 新FUSED | 時間短縮（対PACKED） | FUSED / FAKE |
|---|---:|---:|---:|---:|---:|
| 3752 | 99.997 | 217.375 | 177.038 | **18.556%** | 1.770倍 |
| 31131 | 99.384 | 216.611 | 176.612 | **18.466%** | 1.777倍 |

従来PACKEDに対する速度比は1.228／1.226倍。wall計時でも短縮率18.553%／18.463%と同方向だった。PACKED/FUSEDの実行順を逆にした両冊で短縮した。

**モデル全体が9倍速くなった、という結果ではない。** 展開だけの改善が計測したモデル経路へ届いた幅は約18.5%であり、FAKEに対してはまだ約1.77倍の時間がかかる。FAKEも量子化後の値を保持する方式で、元の非量子化NATIVEに対する新しいPPL試験ではない。

## 4. 同じ実KVで測った部品比較

各冊の従来PACKED終端からslot0／95／191のK/Vを保存し、計12標本を使った。圧縮768 token＋BF16 tail80 token。各経路3round、各5warmup＋20測定、循環順は取得前固定。

下表の時間は、標本ごと3round中央値を求めた上で、その12標本の中央値。短縮率は同じ標本内の比から求めた中央値と最小〜最大であり、時間中央値の単純な比とは少し異なる。単位はms/呼出し。

| 経路 | 従来EAGER | 新FUSED | 対応する短縮率の中央値 | 標本間の範囲 |
|---|---:|---:|---:|---:|
| 3bit codeの展開のみ | 0.120978 | 0.012819 | **89.396%** | 88.793〜89.533% |
| 展開＋NSN全復元＋即時検査 | 0.258562 | 0.150118 | **41.930%** | 41.684〜42.289% |
| 全復元＋即時検査＋tail連結 | 0.265274 | 0.155116 | **41.489%** | 41.047〜42.214% |

三つの比較は各12/12標本で短縮。展開単体の速度比中央値は9.430倍（8.923〜9.554倍）。wall計時の短縮率中央値はそれぞれ88.822%／41.797%／41.349%だった。全標本・全roundの中央値と範囲は独立再集計JSONに保存した。

!PU01の部品とモデル時間 (`reports/figures/pu01-unpack-model-times.png`)

部品測定は同じ入力を繰り返すwarm-cache診断であり、時間を加算・乗算してモデル実行時間を推定しない。12標本は2冊・同じモデルから得た関連するKVで、36の独立実験ではない。範囲は信頼区間ではなく、設計どおりp値・CIは出していない。GPU内cache、演算起動数、一時bufferの各寄与は今回分離していない。

## 5. メモリ使用量は維持

両冊とも従来PACKEDとFUSEDの終端storageは完全に同じだった。

| 測定対象 | FAKE | PACKED | FUSED |
|---|---:|---:|---:|
| 実GPU KV storage | 1,272.00 MiB | 381.84375 MiB | 381.84375 MiB |
| 全forward/post allocatedピーク | 4,087.80 MiB | 4,026.66 MiB | 4,026.66 MiB |
| decode/post allocatedピーク | 4,087.80 MiB | 3,228.23 MiB | 3,228.23 MiB |
| forward/post reservedピーク | 4,484 MiB | 4,646 MiB | 4,646 MiB |

KV storageはFAKE比**69.981%削減**。内訳はcode216.00、metadata45.00、header0.84375、tail120.00 MiB。GPU常駐bufferを数えた値である。モデル重み等を含む全allocatedピークの削減は約1.50%、decode/postピークは約21.03%で、GPU総使用量が70%減ったという意味ではない。reservedピークはFAKEより約3.61%増えた。表のピークは計測対象forward/postの範囲で、全プロセスを外部モニターした最大値とは異なる。

## 6. 結果の解釈と測定範囲

今回は、**保存形式・復元値・即時検査を保った整数展開の局所変更が、部品と短いモデル経路の両方を速くした**。旧PD01の診断から、実際のモデル経路に反映された変更へ一段進んだ。

既知2冊・同じGPU・単一モデル・短い終端長で、各冊各方式のモデル軌道は一回である。量子化そのものの品質、別GPUや長文脈での速度、長期の安定性まで広げる材料は今回増やしていない。旧PK01の最終INCONCLUSIVEと、元のBF16 full/incremental P0未達は保持する。

## 7. 証拠と再計算

- 原ZIP (`runs/loop-kv-pu01-20261001-071446-c65ca1a2.zip`)：SHA256 `1f20a31a394fd781b75ead5aac9566b929bb8547a75dbe0776908bf51e67f641`
- 原runの状態 (`runs/loop-kv-pu01-20261001-071446-c65ca1a2/run_status.json`)／原集計 (`runs/loop-kv-pu01-20261001-071446-c65ca1a2/summary.json`)
- 独立再集計 (`throughline/audits/loop-kv-pu01-results-v0.1-2026-10-01/offline-verification.json`)／再集計コード (`scripts/analyze_loop_kv_pu01_return.py`)
- 図のコード (`scripts/plot_loop_kv_pu01_return.py`)／改変標本の拒否 (`throughline/audits/loop-kv-pu01-results-v0.1-2026-10-01/negative-control-report.json`)

原78ファイルだけをrunへ保存し、派生解析を監査フォルダへ分離した。原ZIP・原採点・取得前カードの埋込版・元statusを保持。今回ローカルのモデルロード、GPU実測、新しい外部データ取得は0件。別実装による照合であり、別人による第三者監査とはしない。

```sh
python3 scripts/analyze_loop_kv_pu01_return.py \
  runs/loop-kv-pu01-20261001-071446-c65ca1a2 \
  /tmp/pu01-offline-recheck.json
```
