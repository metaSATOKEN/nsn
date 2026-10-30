# LOOP-KV-KV01 取得前カード v0.1

2026-09-30。ユーザー承認：「読んだw次のKだけ／Vだけを圧縮する小比較いいと思うw 設計と実装コードまで書いてちょー！w」。設計・コード準備が承認範囲。ユーザーがColabで実行する。

実装変更の前に固定：r8・NSN3・PREFIX_ONLY、既存8冊×NATIVE/K_ONLY/V_ONLY/KV＝32新規軌道。native/KVも再測定、過去値は合算しない。全層／周回を対象とし、codecは凍結する。

問いD=NLL(K_ONLY)−NLL(V_ONLY)、I=NLL(KV)−NLL(K_ONLY)−NLL(V_ONLY)+NLL(NATIVE)。方向の事前予想なし。各8冊平均±0.005でK_LARGER/V_LARGER、SUPERADDITIVE/SUBADDITIVE、幅内を対称に分類。平均と冊bootstrap10,000回・seed2026093002の95%区間を残す。欠けた必要条件があれば固定分類NOT_EVALUABLE。3介入の品質目安5%は副評価で、改善を要求しない。

経路ゲート：r8のnative15 case・45比較＋従来KVと新KVの4 case・4比較。全49比較bitwise一致／RNG不変とfixture合格で進む。旧P0の未達は保持。全192 slotで介入直後に未選択側とprotected tailの不変を検査。後続計算による未選択KVの変化は許す。

上限：準備30分、gate30分、比較60分、case180秒／軌道900秒、10GiB。数値異常は保存して次条件、装置・資源・時間不良は停止。再試行・結果を見て追加なし。

一次証拠：保存token別値、side別介入記録、同一実行native参照、固定点分布、全KV比較hash、コード／材料hash、Colab環境。CPU合成テストは実GPU結果の代替ではない。

検証予定：side誤指定・未選択側の誤変更・tail誤変更・旧KVとの不一致を拒否し、ゼロ変化は正常受理。missing/数値異常を0損失へ補わない。mock ColabでSecret不良・watchdog・ZIP回収を点検する。

取得後欄：未実施。符号にかかわらず固定範囲で閉じる。
