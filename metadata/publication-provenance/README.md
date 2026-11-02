# 既公開NSNと統合稿の対応

Kentaro Sato, *Norm Separation is Necessary: A Minimalist Calibration-Free INT3 Recipe for KV Cache Quantization*, v1.0.0 (2026). [公開版 DOI](https://doi.org/10.5281/zenodo.19724817)

統合稿の付録D.5で再集計した既公開結果の範囲を示す。

|公開版の箇所|採用した対応|統合稿での扱い|
|---|---|---|
|§5.1、表1|8モデルの対称／非対称NSNの品質|原保存PPLから再集計。表示丸めの差は元値に従う|
|§5.2、表2|A/B/C/Dの四方式、閾値内3/3/5/8|統合稿表4。完全要因計画ではないため構成全体の差として述べる|
|§5.3、表2b|K/Vノルム変動とB悪化の対応|付録A.1。相関と因果を分ける|
|§5.4、Phase 26|実packing、模擬QDQとの差|§5.1。整数の可逆性・復元値・PPLを分ける|
|§6.1–6.2、Phase 27|サイズ・bit幅による境界、復元実装の計時|§4と§5の表。保存値・分母・計時範囲を保持|
|§7、Appendix C|GemmaのB方式、Kのみ／Vのみの診断|§4.5と表A3。約87%を因果寄与率にしない|

公開PDFの同一性と書誌を照合するための記録を収録している。[PDFの識別情報](pdf-identity.json)・[Zenodo記録](zenodo-record.json)・[DataCite記録](datacite-doi.json)
