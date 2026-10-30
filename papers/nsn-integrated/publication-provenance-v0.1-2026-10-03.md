# 旧NSN公開版の照合と統合稿での採用範囲

2026-10-03／v0.1。日本語草稿v0.3の文献［12］と付録D.5に対応する。著者から提示されたURLを起点に公開記録を取得し、書誌とファイルを照合した。

## 書誌と版

|項目|公開記録の値|
|---|---|
|著者|Kentaro Sato（登録表記：Sato, Kentaro）|
|題名|Norm Separation is Necessary: A Minimalist Calibration-Free INT3 Recipe for KV Cache Quantization|
|指定版|v1.0.0|
|版DOI|[10.5281/zenodo.19724817](https://doi.org/10.5281/zenodo.19724817)|
|親DOI|10.5281/zenodo.19724816。今回の指定版引用には使わない|
|資料種別|Zenodo preprint|
|年|2026|
|Zenodoのpublication_date／DataCiteのIssued|2026-04-24|
|Zenodo記録のcreated|2026-05-22T04:44:31.462334+00:00|
|PDF表紙とローカルMDの日付|2026-05-22|
|公開PDFの利用条件|CC BY 4.0。統合稿のライセンスを自動的に決めるものではない|

登録上の発行日とPDF上の日付は異なるため、参考文献は確認できた年と版を記す。異なる日付を同一の出来事の日付へ直さない。Zenodo APIとDataCite APIの題名・著者・年・版・DOIは一致した。

## 公開ファイルと手元の対応

- 公開ファイル：`nsn.pdf`、525,735 byte。
- 公開MD5：`a748110e7f25254f1ee7cba7d037c9a9`。ダウンロードしたbytesから一致を確認。
- SHA256：`e68359fcc4c9de43dc43e631be7086caae1fd416de473bcf4f70cdcc5560f06f`。
- 比較対象：[ローカル保存PDF](https://doi.org/10.5281/zenodo.19724817)。SHA256だけでなくbyte列も完全一致。
- 本文の抽出と表紙・表1／2のレンダリングを点検した。PDFは29ページ。レイアウト全体の新規校正は今回の目的ではない。

公開メタデータの取得はweb取得ツールでは失敗し、公開APIの読み取りに切り替えて成功した。取得したメタデータ・原PDF・照合結果は[今回監査](../../metadata/publication-provenance/)へ保存した。GitHub検索結果のみでZenodoの公開を認定したものではない。

## 内容対応と再解析の区別

|公開版の箇所|採用した対応|統合稿での扱い|
|---|---|---|
|§5.1、表1|8モデルの対称／非対称NSNの品質|原保存PPLから再集計。表示丸めの差は元値に従う|
|§5.2、表2|A/B/C/Dの四方式、閾値内3/3/5/8|統合稿表4。完全要因計画ではないため構成全体の差として述べる|
|§5.3、表2b|K/Vノルム変動とB悪化の対応|付録A.1。相関と因果を分ける|
|§5.4、Phase 26|実packing、模擬QDQとの差|§5.1。整数の可逆性・復元値・PPLを分ける|
|§6.1–6.2、Phase 27|サイズ・bit幅による境界、復元実装の計時|表5と表7。保存値・分母・計時範囲を保持|
|§7、Appendix C|GemmaのB方式、Kのみ／Vのみの診断|§4.4と表A3。約87%を因果寄与率にしない|

ローカルMDと公開PDFの該当節・値の対応を確認したが、異なる形式の全文byte同一を主張するものではない。また、公開PDFとの一致は、ローカルの全実験ログが公開アーカイブに含まれることの証明ではない。元値への入口は統合稿付録Dに示す。

SADPは著者申告どおり未公開。Looped Transformer系列は既公開NSNに含まれない追加評価で、校了済みの研究内稿と原runを入力資料として統合する。統合稿の執筆時にモデル実験を増やしたという意味ではない。

## 照合に使った一次情報

- [Zenodo record](https://zenodo.org/records/19724817)
- [Zenodo API](https://zenodo.org/api/records/19724817)
- [DataCite DOI metadata](https://api.datacite.org/dois/10.5281/zenodo.19724817)
- [公開PDF](https://zenodo.org/api/records/19724817/files/nsn.pdf/content)
