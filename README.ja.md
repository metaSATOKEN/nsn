# NSN：ノルムと方向を分けるKV量子化

Kentaro Sato · 独立研究者 · [info@metaclan.jp](mailto:info@metaclan.jp)

LLMは、それまで読んだ情報をattention用の「KVキャッシュ」に保存します。NSNは、このベクトルの**大きさを別に保存し、方向を少ないビットで表す**圧縮方法です。非対称NSNはチャネル別の最小値と刻み幅を使い、復元した方向を再正規化します。3bit専用ではなく、精度を重視した5bit設定も評価しています。

このリポジトリは、**品質を保てる設定 → 実際の保存量 → 圧縮した情報を使う費用**を一続きに検証した研究資料です。キャッシュが小さいこと、GPU全体のメモリが減ること、推論全体が速いことを、それぞれ測っています。

まずは[日本語の統合論文 v1.0](papers/nsn-integrated/NSN-2026-10-29-JA.md)／[English v1.0](papers/nsn-integrated/NSN-2026-10-29-EN.md)へ。旧NSNを先に読む必要はありません。[再帰型モデルの詳細稿](papers/loop-kv/README.md)も収録しています。

## 主な結果

- 四方式の主条件比較では、非対称NSNの3bit設定が8モデルすべてでPPL悪化5%未満でした。構成全体の比較であり、条件を変えたときの限界も論文に示しています。
- SSM＋attentionやMoEを含む4モデルでも、attentionのKVを全窓QDQで評価した5bit設定が品質の診断目安を満たしました。
- 実packed表現をOuroとNanbeigeへ接続し、終端KVをそれぞれ67.68%、継続評価では64.39%削減しました。各4文章、Ouroは255予測、Nanbeigeの追補は256予測です。
- 復元をattentionに組み込むと圧縮経路は短縮しました。一方、Nanbeigeの追補では初期圧縮・更新費用を含めると検査付き非圧縮とほぼ同時間でした。

品質はPPLによる診断基準として扱います。自由生成や一般的な応答品質まで保証する意味ではありません。

公開用の実験コードは[展開済みのPython・設定ファイル](experiments/README.md)です。圧縮OneCellを貼り付ける必要はありません。手元のColab用OneCellは元の研究フォルダに保持しています。

## 統合論文に沿った配置

| 内容 | 場所 |
| --- | --- |
| NSNの参照実装 | [src/nsn](src/README.md) |
| 四方式・bit幅などの構成比較 | [experiments/configuration](experiments/configuration/README.md)／[data/configuration](data/README.md) |
| 複合モデル・長さ・適用位置の比較 | [experiments/applicability](experiments/applicability/README.md)／[data/applicability](data/README.md) |
| 保存量・復元時間・融合部品 | [storage](experiments/storage/README.md)／[kernels](experiments/kernels/README.md) |
| Loopedモデルでの実測 | [AC08・NB05・NB06](experiments/README.md) |

原結果の内容は保持し、旧研究のフォルダ名を使わずに整理しました。旧論文本文は公開本体から外し、既公開NSNへの引用を残しています。過去の記録に残る元の識別子と、現在の保存先は対応表で照合できます。

## 読む・検証する・再計算する

| やりたいこと | 入口 |
| --- | --- |
| 論文と図を見る | 上記の現行稿と `papers/nsn-integrated/figures/` |
| 手元のファイルが配布物と一致するか確認 | [RUNNING.md](RUNNING.md) のhash検証 |
| 大きなログ・保存配列を開く | [データ範囲](DATA_COVERAGE.md) と別配布ZIPの復元 |
| 保存値から再集計・原稿を再生成 | [再現手順](REPRODUCIBILITY.md) の別作業コピー |
| GitHubへ公開する | [公開前チェックリスト](PUBLISHING.md) |

研究ルートの `.venv/bin/python` を使います。単独コピーも、そのコピーのルートに環境を用意します。

```sh
.venv/bin/python -B tools/verify_package.py
```

GitHub本体には原稿・図・コード・集計と読めるログを置き、保存配列と大きなログは別配布ZIPへ分けています。元の入力や結果を変更する処理ではありません。

統合版は[https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn)に収録し、**17本のデータZIPとSHA256SUMS（ZIP計約6.30 GB）を[integrated-v1.0 Release](https://github.com/metaSATOKEN/nsn/releases/tag/integrated-v1.0)に添付**しています。ZIPはすべて同じディレクトリへ、改名せず取得してください。GitHubが自動生成するソースアーカイブには、この別配布データは入りません。[再現手順](REPRODUCIBILITY.md)から照合・復元できます。**著者の公開判断まではリポジトリをプライベートに保つため、閲覧権限が必要です。** [旧版とライブラリの案内](LEGACY.md)。コードはApache-2.0、著者作成の論文・図・データはCC BY 4.0、第三者資料は元の条件を保持します。[利用条件](LICENSES.md)。現在のPDFは表紙の旧メールアドレス訂正後に追加するため、今回は訂正済みMDを収録しています。

内部の台帳・運用履歴は公開対象から除外しました。論文に必要な実験条件は `plans/prespecified/`、数値検証は [data/validation](data/validation/README.md) に収録しています。
