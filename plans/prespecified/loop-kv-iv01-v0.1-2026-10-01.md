# IV01取得前カード

2026-10-01 / v0.1。DC01の18実標本を固定して入力検査を集約する。LEGACY／TORCH_GROUPED／TRITON_GROUPED、CHECK_ONLY／PACKED_PIPELINEの108組合せ。正常8・非有限36・構造20の合計64fixture×3方式を先行する。

予想は判定・出力一致、新Triton集約によるCHECK_ONLY短縮とPACKED_PIPELINEへの短縮の残存。中間torch条件の方向と効果量は未予測。全108初回＋各5warmup後、主3round×20＝324区間。速度の方向で完了判定を変更しない。欠損・非有限・gate不一致は部分保存付きINCONCLUSIVE、cache変化は別フラグで保持。

各呼出しで同じ六配列と構造を検査し、異常時にはattentionへ進む前に停止。host確認も計時内。入力検査の削除・V復元融合・モデル速度／PPLへの昇格は行わない。GPU実測前にコード・素材hash・判定条件を固定する。

元DC01 ZIPを/contentに配置、Drive／HF_TOKEN／モデルなし、1セル・ZIP自動DL。準備10＋実験10分、条件120秒、照合2段とZIP回収各120秒、512MiB。[設計](../../plans/loop-kv-iv01-design-v0.1-2026-10-01.md)。
