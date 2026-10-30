# PK01結果：実3bit保存の容量・予測・速度と、数値照合の未達

2026-10-01。対象：`loop-kv-pk01-20261001-060906-22517c43`（worker v0.1.1）。

**本比較6軌道・1,530予測は完了し、実packed cacheは疑似圧縮と同じ予測を保ったまま、実保存量を67.68%減らした。現実装のdecode時間は約2.17倍だった。** 最終の独立NLL照合が事前許容幅に届かなかったため、原runの正式状態は`INCONCLUSIVE`のまま保存する。以下では、取得済みの測定事実と、その後の数値診断を分ける。

## 1. 何を比較したか

同じモデル・入力・量子化値について、保存の仕方を変えた。

| 方式 | KVの値 | GPUに置く形 |
|---|---|---|
| NATIVE | 元のBF16 | BF16 tensor |
| FAKE | NSN3で量子化して復元した値 | 復元後のBF16 tensor |
| PACKED | FAKEと同じNSN3値 | 3bit code・付帯情報・未圧縮tail。利用時に復元 |

Ouro-1.4B固定版、BF16、8周、192 slot。既知2冊（3752／31131）、入力1,024 token、prefill768、採点位置769〜1023の255予測／軌道。終端cacheは1,023 tokenで、896 tokenを圧縮し、127 tokenをBF16で保持した。両冊でFAKE/PACKEDの実行順を逆にした。材料数は2冊で、軌道やtokenを独立材料数へ水増ししない。

[設計](../plans/loop-kv-pk01-design-v0.1-2026-10-01.md)／[取得前カード](../throughline/prereg/loop-kv-pk01-v0.1-2026-10-01.md)。原BF16 full/incremental P0の未達は継続して保持する。

## 2. 実行と検証の状態

| 項目 | 結果 |
|---|---|
| CPU/CUDA合成fixture | 8/8 PASS |
| 先行モデルゲート | 6軌道・390予測完了、4経路比較すべて一致 |
| 本比較 | 6/6軌道・1,530/1,530予測、欠測0、すべて有限 |
| workerの保存量判定 | `PASS_STORAGE` |
| 最終launcher判定 | `INCONCLUSIVE`：別計算NLLの許容幅未達 |
| オフラインhash・予定・原値照合 | 56ファイルhash・8埋込ソース・全1,932回のlogits記録を照合 |

実GPUはRTX PRO 6000 Blackwell Server Edition、torch 2.11.0+cu128。worker実行約6.79分。初回のNumPy未定義エラーはこの実行では発生せず、モデル実行・ゲート・主比較まで進んだ。最初の準備停止runとは別に保存した。

再集計器は返却workerをimportせず、保存token・BF16 logitsのbit列・タイミング列・容量記録から照合した。GPUメモリは保存済み実測値の再集計であり、手元で再測定したものではない。hashを更新した別コピーの誤target・誤容量・FAKE/PACKEDのKV不一致をすべて拒否し、原本のNLL未達はそのまま報告する検査も通した。

## 3. 実保存量とGPUピーク

次のメモリ値は両冊で共通。MiB = 2²⁰ byte。削減率の比較元はFAKE。

| 指標 | NATIVE | FAKE | PACKED | PACKEDの変化 |
|---|---:|---:|---:|---:|
| 終端KVの実unique storage | 1,534.50 MiB | 1,534.50 MiB | 495.98 MiB | **67.68%削減** |
| 全forward/post工程のallocatedピーク | 4,312.87 MiB | 4,313.74 MiB | 4,026.66 MiB | 6.66%削減 |
| decode/post工程のallocatedピーク | 4,312.87 MiB | 4,313.74 MiB | 3,298.60 MiB | **23.53%削減** |
| 全forward/post工程のreservedピーク | 4,484.00 MiB | 4,484.00 MiB | 4,748.00 MiB | **5.89%増加** |

PACKEDの495.98 MiBは、code 252.00、metadata 52.50、header 0.98、保護tail 190.50 MiBの合計。実tensorのunique storageと計算量が一致し、全bufferはGPUにある。CPUへKVを逃がした削減ではない。

保存されるKVは約3分の1になった。しかしモデル本体・未圧縮の初期読み込み・一時展開もGPUメモリを使うため、実行全体のピークが同じ割合で減るわけではない。PACKEDではprefillがallocatedピークを決めた。

`allocated`はPyTorchが現在のtensor等に割り当てた領域、`reserved`はallocatorが確保している領域。今回reservedは増えている。固定snapshotの監査でKVを展開する処理もallocatorの履歴へ影響し得るため、この計測から「GPUの全メモリ指標が減る」とは結論しない。表のピークは事前設計に従うforward/post区間で、監査処理を含めた全プロセスの最大値ではない。各軌道終了後のallocated/reservedは全6件で開始前baselineへ戻った。

## 4. 予測の一致と品質

両冊とも、**FAKEとPACKEDの全256回の全語彙logitsがbit単位で一致**した。固定8時点の全192 slot・K/V digestも一致し、採点した255件のNLLも完全一致した。KVについて照合できる範囲は保存された固定時点であり、全decode時点のKV実体を保存したわけではない。

| 冊 | FAKE/PACKEDのlogits | 元の採点によるNATIVE比PPL悪化 | 事後FP64再計算 |
|---|---|---:|---:|
| 3752 | 完全一致 | +0.287813% | +0.287822% |
| 31131 | 完全一致 | +1.219769% | +1.219769% |

この条件では、実packed化によるFAKEへの追加品質損失は観測されなかった。NATIVEからの差は量子化に伴う差として残る。2冊の既知材料の接続確認であり、新規材料への品質推定や統計的な一般化は行わない。

## 5. 速度との交換条件

初期decode8ステップを除き、残る247ステップのforward＋post（flush込み）を集計した。CPU採点・logits保存・固定snapshotの監査時間は含めない。

| 冊 | NATIVE | FAKE | PACKED | PACKED / FAKE |
|---|---:|---:|---:|---:|
| 3752 | 97.14 ms/token | 99.37 ms/token | 215.80 ms/token | 2.172倍 |
| 31131 | 97.17 ms/token | 99.68 ms/token | 216.67 ms/token | 2.174倍 |

CUDA Eventとwall timeの双方で同じ傾向だった。**小さく保存することはできたが、現実装ではその都度展開して読む処理が高くついた。** 復元、連結、codec内の検査等を含む参照実装全体の時間であり、内訳別の原因の大きさは未測定。融合kernelや最適化後の速度を表す結果ではない。

## 6. なぜ最終表示は未確定か

直接原因は独立scorerの照合。workerはCPU上で`torch.log_softmax(logits.float())`によりFP32のNLLを保存し、独立照合器は同じBF16 logitsからFP64のlog-sum-expを計算する。事前の絶対許容幅は**0.000020**だった。

全1,920採点を調べると、gateは390件中0、本比較は1,530件中12件が許容幅を超えた。12件にはFAKE/PACKEDで同じ値が現れる重複を含む。最大差は**0.0000237573**。最初の停止位置は3752／NATIVEのtarget817、差0.0000205733であり、PACKED固有のエラーではない。

| 本比較軌道 | 超過数 / 255 | 最大絶対差 |
|---|---:|---:|
| 3752 NATIVE | 3 | 0.0000205733 |
| 3752 FAKE | 3 | 0.0000237573 |
| 3752 PACKED | 3 | 0.0000237573 |
| 31131 NATIVE | 1 | 0.0000211680 |
| 31131 PACKED | 1 | 0.0000211498 |
| 31131 FAKE | 1 | 0.0000211498 |

超過12件をFP64の`math.fsum`による和とCPU PyTorch FP64でも照合した。NumPy FP64との差はそれぞれ最大1.78×10⁻¹⁵、6.17×10⁻¹⁴だった。一方、同じ原logitsを手元のCPU PyTorch FP32で採点してもColabの保存値とは一致しなかった。**FP32採点とFP64参照計算の数値差という説明と整合する**が、ColabのCPU命令・kernel・依存環境のどれが差を生んだかは切り分けていない。

保存された原logitsと各時点digestに食違いは見つからず、FP64でPPLを再計算しても§4の傾向は変わらなかった。この追加計算は事後診断。元NLLを置換せず、2e-5の閾値を緩めず、`INCONCLUSIVE`を`PASS_STORAGE`へ書き換えない。workerの`PASS_STORAGE`は保存量・FAKE/PACKED一致の集計状態であり、最終の全検証通過とは区別する。

## 7. 到達点と次の一手

取得前の「復元一致と常駐量削減」の予想に沿う測定が得られた。速度向上は予想・合格条件に含めておらず、今回の遅さも結果として残す。数値照合を含む正式な検証完了は未達である。

**次の推奨は、GPU実験を追加する前に採点経路をFP64へ揃えるオフライン修正。** 保存済みlogitsと合成fixtureで、別実装との照合・誤target／意図的なNLL改変の拒否を点検する。観測した最大誤差に合わせて許容幅を広げる方法は採らない。修正を次回実験に適用する場合は版と取得前条件を新たに固定する。

その後の問いは「どの復元・連結処理が時間を使っているか」。速度内訳の小診断または最適化候補の設計へ進める。長cache・新材料への拡張はその目的が定まった段階で検討する。今回の6軌道を成功まで増やしたり、過去LC01/LC02と独立反復として合算したりしない。

## 8. 証拠と再計算

- [原ZIP](../runs/loop-kv-pk01-20261001-060906-22517c43.zip)：SHA256 `b21f6cb661e657481bb7459135b72d737e9381654b94ec89c7033597907f539f`
- [原runの最終状態](../runs/loop-kv-pk01-20261001-060906-22517c43/run_status.json)、[worker集計](../runs/loop-kv-pk01-20261001-060906-22517c43/worker_summary.json)、[元の照合停止](../runs/loop-kv-pk01-20261001-060906-22517c43/recovery_verification.log)
- [独立再集計JSON](../throughline/audits/loop-kv-pk01-results-v0.1-2026-10-01/offline-verification.json)：全token FP64参考値・超過12件・容量・時間・ゲートを収録
- [再集計コード](../scripts/analyze_loop_kv_pk01_return.py)、[改変標本検査](../throughline/audits/loop-kv-pk01-results-v0.1-2026-10-01/negative-control-report.json)

展開run内は原57ファイルのみを保ち、派生分析は監査フォルダへ分離した。再計算はPython＋NumPy、超過行の補助照合にはCPU PyTorchを使用する。モデル・GPU・通信は不要。

```sh
python3 scripts/analyze_loop_kv_pk01_return.py \
  runs/loop-kv-pk01-20261001-060906-22517c43 \
  /tmp/pk01-offline-recheck.json
```
