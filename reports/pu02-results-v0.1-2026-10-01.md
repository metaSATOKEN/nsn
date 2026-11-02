# PU02結果：圧縮更新を3回通しても、値一致と約19%の時間短縮を維持

2026-10-01 / v0.1。LOOP-KV-PU02は**COMPLETE_COMPARISONで完了**した。PU01の整数展開kernelを固定して後続区間を80から255予測へ延ばしたところ、従来PACKEDとの予測値一致を保ち、モデル経路の時間が2冊で19.394%／18.706%短くなった。後半の区間でも短縮は続いた。終端の実KV storageはFAKE比67.678%削減を維持した。

## 1. 第一段階で確かめたこと

同じ量子化値を、FAKEはBF16で保持し、PACKEDは3bitで保存して従来処理で展開し、FUSEDは同じ保存物をPU01の単一Triton kernelで展開する。今回の変更は後続区間の長さであり、kernel、浮動小数点復元、即時有限値検査、tail連結、attention、CPU FP64採点は維持した。

Ouro-1.4Bの固定revision `fcfcfbc429c7b10ee73dfc718033351d8936d2a8`、BF16・8周、既知2冊3752／31131。prefill768からcache1023まで進め、各条件255予測を採点した。decode中のcache長832／896／960で圧縮範囲を更新する。同じGPUのRTX PRO 6000 Blackwell Server Edition、torch 2.11.0+cu128、Triton 3.6.0で実測した。

固定設計 (`plans/loop-kv-pu02-design-v0.1-2026-10-01.md`)／取得前カード (`throughline/prereg/loop-kv-pu02-v0.1-2026-10-01.md`)。取得前の予想は、両冊でFUSEDがPACKEDより短時間で、値と保存形式は同じ、というもの。今回の結果はこの予想に沿った。

## 2. 完了と値の照合

| 対象 | 結果 |
|---|---|
| 主比較 | 6/6軌道、1,530/1,530採点、3flush/軌道 |
| 先行モデルゲート | 7軌道・455採点、5比較が合格 |
| 初期検査 | cache8件＋凍結kernel17件＋追加13／14block形状2件が合格 |
| 全forward | 1,998件のraw logitsが有限、予定位置・順序と一致 |
| 独立FP64採点 | 1,985/1,985件が絶対2e-5以内、最大差3.2731×10⁻¹⁴ |
| 保存物 | 原62ファイル・61hash、配布10埋込ファイルが一致 |
| 実行時間 | worker計時482.31秒、約8分2秒 |

両冊ともFAKE／PACKED／FUSEDの全256forwardの全語彙logitsがbit単位で一致し、255の保存NLLも一致した。KVは固定8時点（768、831、832、895、896、959、960、1023）×192slotのK/V digestが一致した。3回目の圧縮更新後と最後の予測まで、同じ値を引き継いでいる。

返却workerをimportしない別集計器で、原BF16 bit列からのFP64採点、token対応、全計測位置、storage内訳、時間集計を再計算した。旧kernelの16入力fixtureと追加2形状はCPUで正解codeのhashを作り直して照合した。GPUでの展開・ゼロ受理・NaN拒否の成立は保存されたゲート記録による。

別コピーを再hashしたうえで、最後のNLL誤記、3回目のflush欠落、最後のtiming欠落、最終KV digestの不一致、後半の時間集計誤記、追加形状の正解hash不一致、storage誤記を入れた7負例はすべて拒否した。原本は受理した。再集計・検査記録 (`throughline/audits/loop-kv-pu02-results-v0.1-2026-10-01`)。

## 3. 長い区間と後半でも時間短縮が続いた

初期decode8回を除く247回のforward＋postを主集計にした。3回の圧縮更新を含むCUDA Event平均で、単位はms/token。

| 冊 | FAKE | 従来PACKED | FUSED | 対PACKED時間短縮 | FUSED / FAKE |
|---|---:|---:|---:|---:|---:|
| 3752 | 100.408 | 218.082 | 175.788 | **19.394%** | 1.751倍 |
| 31131 | 100.229 | 217.425 | 176.753 | **18.706%** | 1.763倍 |

wall計時の短縮率も19.391%／18.703%だった。各冊でPACKED/FUSEDの実行順を逆にした固定順でも、両方で短縮した。

事前に分けた早い区間と後ろの区間も、同じ計測列から集計した。

| 冊 | 早い区間：72回・1flush | 後ろの区間：175回・2flush |
|---|---:|---:|
| 3752：対PACKED時間短縮 | 19.352% | **19.411%** |
| 31131：対PACKED時間短縮 | 18.629% | **18.738%** |

**初期区間だけで生じた短縮ではなく、圧縮blockが増えた後半にも同程度の差が残った。** 早い／後ろは一本の軌道内の区間であり、独立した短長二試験ではない。

!PU02のモデル時間と区間別短縮 (`reports/figures/pu02-long-model-times.png`)

計測はforwardとpostの範囲。復元中の即時検査は含み、モデルロード・コンパイル・CPU採点・snapshot監査・保存は外側にある。FAKEも量子化後の値を持つ対照なので、この表は元の非量子化モデルに対する品質や速度の比較ではない。

## 4. 保存容量とGPU使用量

両冊でPACKEDとFUSEDの終端保存量・内訳は同じだった。終端は圧縮896token＋BF16 tail127token。

| 測定対象 | FAKE | PACKED | FUSED |
|---|---:|---:|---:|
| 実GPU KV storage | 1,534.50 MiB | 495.984375 MiB | 495.984375 MiB |
| 全forward/post allocatedピーク | 4,313.74 MiB | 4,026.66 MiB | 4,026.66 MiB |
| decode/post allocatedピーク | 4,313.74 MiB | 3,298.60 MiB | 3,298.60 MiB |
| forward/post reservedピーク | 4,484 MiB | 4,748 MiB | 4,748 MiB |

実KV storageはFAKE比**67.678%減**。packed内訳はcode252.00、metadata52.50、header0.984375、tail190.50 MiB。テンソルの計算上のサイズと、重複を除いた実storageサイズが一致し、保存形式の予測値とも一致した。

モデル重み等を含むallocatedピークの削減は全計測工程で6.655%、decode/postで23.533%。reservedピークは5.888%増えた。したがって、KV保存量の削減とGPU全体の使用量は別に読む。表は計測した工程のピークである。

PU01の約70%減から割合が変わったのは、終端長と未圧縮tailの比率が異なるため。同じ終端長のPACKEDとFUSEDを比べると差はなく、kernel変更で保存効率を失った結果ではない。

## 5. 到達点と第二段階

第一段階では、**整数展開の局所変更が、3回の圧縮更新を通したモデル経路でも値・検査・容量を保ち、約19%の時間短縮を維持した**。後続区間を延ばした今回の範囲まで、PU01の成立範囲を広げられた。

それでもFUSEDはFAKEより約1.75〜1.76倍遅い。

今回の完了範囲は既知2冊・同一モデル/GPU・各条件1軌道、最大cache長1023。旧PU01と独立Nとして合算せず、p値・CIは出していない。量子化自体のPPL改善、別GPU、未使用材料、さらに長い文脈は別の問いとして残る。旧PK01のINCONCLUSIVEと旧BF16 full/incremental P0未達は保持した。

## 6. 証拠と再計算

- 原ZIP (`runs/loop-kv-pu02-20261001-074838-a39a8745.zip`)：SHA256 `6e145df237d2891489a3ec9e7408cd642242519c0d3041a75cc8905564f516c3`
- 原状態 (`runs/loop-kv-pu02-20261001-074838-a39a8745/run_status.json`)／原集計 (`runs/loop-kv-pu02-20261001-074838-a39a8745/summary.json`)
- 独立再集計 (`throughline/audits/loop-kv-pu02-results-v0.1-2026-10-01/offline-verification.json`)／再集計コード (`scripts/analyze_loop_kv_pu02_return.py`)
- 改変標本の拒否 (`throughline/audits/loop-kv-pu02-results-v0.1-2026-10-01/negative-control-report.json`)／図のコード (`scripts/plot_loop_kv_pu02_return.py`)

原62ファイルをそのままrunに保存し、派生解析を監査フォルダへ分けた。元ZIP、採点値、取得前カードの埋込版、原statusは保持。今回手元ではモデルロード・GPU実測・新外部取得を行っていない。別集計器の照合であり、別人による第三者監査ではない。

```sh
python3 scripts/analyze_loop_kv_pu02_return.py \
  runs/loop-kv-pu02-20261001-074838-a39a8745 \
  /tmp/pu02-offline-recheck.json
```
