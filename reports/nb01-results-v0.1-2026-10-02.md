# NB01結果 — Nanbeigeの小接続確認が成立

2026-10-02／v0.1。**PASS_CONNECTION。予定27 forwardを完了し、公式反復・観測用cache接続とも数値がbit一致した。** 配布前のNumPy検証器による手元の再計算も、返却summaryと一致した。

## 今回確認できたこと

固定Nanbeige4.2-3B-Baseを実GPUでロードし、公式cacheの更新に記録を添えても、同じ入力から同じ予測・KVを得られた。Ouroで使った圧縮方式を移すための、最初の接続点が成立した。

|取得前の条件|観測結果|判定|
|---|---|---|
|H1：公式経路の反復 N0↔N1|9位置×全語彙1,495,296 logit値のbit差0。9時点×88 K/V＝792個のcache hashも一致|成立|
|H2：公式 N0↔観測用 A0|全logit・792個のcache hashが一致。終端raw KVの12,255,232個のBF16値もbit差0|成立|
|H3：周回別の保存構造|22層×2周＝44 slot、各forwardでslot 0〜43を各1回、総396 update。終端各K/Vは[1,8,136,128] BF16|成立|
|H4：小容量で完了|27/27 forward、欠測0。証拠64.919 MiB、ZIP45.465 MiB|成立|

N1のraw KVは設計どおり保存していないため、N0/N1のcache比較は各時点のhashによる照合。N0/A0の終端はrawも直接比較した。中間状態の全要素を手元で復元した、という意味には広げない。

## 取得範囲と環境

- 返却run：`loop-kv-nb01-20261002-130959-b8c226de`。
- モデル：`Nanbeige/Nanbeige4.2-3B-Base`、revision `4a38e817c14b9f2c4b69b1e05cbda90395872f6b`。
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition。BF16／SDPA、Q48 head・KV8 head、dimension128、既定2周。実モデルの記録上のparameter countは4,169,800,704。これは配布名「3B」と区別したロード時の実数。
- torch2.11.0+cu128／Transformers4.45.1／tokenizers0.20.3。主要pin・CUDA/BF16検査成立、loader環境の継承も記録されている。
- 固定短文は363 tokenとなり、先頭137 tokenを重みロード前に固定。N0→N1→A0の3系列、各128 token初期入力＋8 decode、予測対象は位置128〜136。
- 次の系列では新しいcacheを使用。生成サンプリング、追加warmup、予定外のmodel forward、量子化、融合kernelは使用していない。

全系列の平均NLLは2.0505270398101465で一致した。9予測位置の接続診断であり、一般文章の品質や圧縮後の品質を評価する標本には数えない。

## 容量と時間の診断

|保存量|実測|上限|
|---|---:|---:|
|主raw（logit＋N0/A0終端KV）|66,964,480 bytes／63.862 MiB|設計見積りと一致|
|ZIP内の全証拠|68,072,824 bytes／64.919 MiB|96 MiB|
|返却ZIP|47,673,462 bytes／45.465 MiB|128 MiB|

モデル重み・venv・Hub cacheは返却物に含まれていない。小さいZIPにする方針も実測で成立した。

環境準備READYまで97.466秒、開始からモデル準備完了まで127.961秒。モデル準備完了から測定・解放を経て検証phaseに入るまで約2.450秒で、測定5分の上限内。記録された各系列時間はN0 0.948秒／N1 0.476秒／A0 0.493秒、各90秒の上限内だった。

これらは全cacheのCPUコピー・hash・ファイル保存も含む接続診断の時間で、初回の準備条件も異なる。3系列の数字をattention速度やadapterの高速化率に転用しない。ZIP生成・ブラウザDL完了までの厳密な所要時間は今回の保存logだけでは再構成しない。

## 証拠の照合

原ZIPは元の場所に保持し、run保存先 (`runs/loop-kv-nb01-20261002-130959-b8c226de`)へ同じ内容を保存した。80ファイル、SHA256一覧79件、配布時の凍結source16件を照合。CRC・相対path・重複・symlink・容量を検査した。入力契約と取得前カードを含むcapsuleは配布版と一致する。

返却コードは実行せず、配布前の`scripts/verify_loop_kv_nb01.py`とcontractを使用した。そのローカルsource自身も配布前のdeployment manifestに照合した。全logitからNLL、終端rawからhashと差分、396 updateの順序・shape・位置・長さを再計算し、summaryが一致した。追加で入力の固定時刻、モデルsource／weightのhash記録、ロード設定、主要packageの版を照合。監査後も返却ファイル79件のhashは不変。

監査記録 (`throughline/audits/loop-kv-nb01-results-v0.1-2026-10-02`)に原ZIPのhash、独立summary、指標と検証器hashを保存。原ZIP SHA256：`f1bf862ffaca43a696d65068a9865f3d96304788381ba0921bd0f883552f908a`。
