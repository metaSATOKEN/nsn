"""Render human-readable tables from extracted evidence, without model execution."""
import argparse, json
from pathlib import Path

def main(data_dir, output):
    t=json.loads((data_dir/'evidence-tables.json').read_text())
    s=json.loads((data_dir/'summary.json').read_text())
    lines=['# NSN統合稿：再集計表 v0.1', '', '2026-10-03。既存の保存値から作った事後の記述表。原実験の判定は変更しない。', '',
           '値の全桁・元JSON内の位置・SHA256は[抽出JSON](data/extraction-v0.1/evidence-tables.json)、計算手順は[集計仕様](protocol-and-decisions-v0.1-2026-10-03.md)を参照。表内の相対PPLは非圧縮比。ppは百分率の差。', '']
    def text(x): lines.extend([x,''])
    def table(headers,rows):
        def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
        lines.append('| '+' | '.join(map(cell,headers))+' |')
        lines.append('| '+' | '.join('---' for _ in headers)+' |')
        lines.extend('| '+' | '.join(map(cell,row))+' |' for row in rows);lines.append('')
    def pct(x):return f'{x:+.3f}'
    def ci(q):return f"[{q['ci_lo']:+.3f}, {q['ci_hi']:+.3f}]" if isinstance(q['ci_lo'],(int,float)) else '算出不能（非有限値）'
    text('## T1. 旧NSNの主要比較')
    text('C＝ノルム分離＋チャネル別対称量子化＋再正規化、D＝非対称版。最大10窓の主条件。初出の集計値から再計算し、後続ファイルへの再録は標本へ加えない。')
    table(['モデル','C 相対PPL %','D 相対PPL %','D <＋5%'],[[x['model'],pct(x['C_percent']),pct(x['D_percent']),'成立' if x['D_under5'] else '超過'] for x in t['legacy_primary']])
    text(f"主条件でDは8/8、Cは5/8モデルが＋5%未満。Cに対する絶対ΔPPLの減少率をモデルごとに計算した平均は{s['primary']['mean_reduction_percent']:.3f}%。これはPPL全体が66%改善したという値ではない。")
    text('## T2. 四方式の相対PPL（%）')
    text('A＝行別対称、B＝チャネル別非対称、C/D＝T1。A/Cは7段階、B/Dは8段階で、補助情報量も異なる。構成全体の比較である。')
    models=list(dict.fromkeys(x['model'] for x in t['legacy_ablation']))
    table(['モデル','A','B','C','D'],[[m]+[pct(next(x for x in t['legacy_ablation'] if x['model']==m and x['method']==a)['percent']) for a in 'ABCD'] for m in models])
    text('＋5%未満のモデル数はA＝3、B＝3、C＝5、D＝8。Phase 19の保存PPLには丸めがあるため、T1の初出値と末尾桁が異なる場合がある。Dが全モデルで最小PPLだったという意味ではない。')
    text('## T3. 主条件の外で確認した境界')
    rows=[]
    for x in t['legacy_boundaries']:
        if 'D' not in x['method']:continue
        if x['run']=='phase20_bf16_disambiguation' and x['label']!='gemma-4-E4B/bf16':continue
        n=f"{x['tokens']//512}窓／{x['tokens']}予測" if isinstance(x['tokens'],int) else '最大10窓（取得窓数未記載）'
        rows.append([x['run'].replace('phase','P'),x['label'],x['method'],pct(x['percent']),n])
    table(['元試験','対象','方式','相対PPL %','分母'],rows)
    text('GemmaのSEG1/2は最大30窓の設定だが、保存token数から求めた実取得は20/22窓。Qwen3.5のfp16/bf16というラベルの二実行は、保存記録ではどちらも実BF16であり、dtype対照には使わない。Phi-3の接続不能は品質不成立と分けて抽出JSONへ保存した。')
    text('## T4. SADP：採用設定での方式群比較')
    text('相対PPL変化を窓ごとに計算した平均。差＝plain−NSN、正はNSN側の変化が小さい。CIは保存された30窓の値と元bootstrap手順から再現した名目95%区間。重複する窓を含み、30独立文章ではない。')
    rows=[]
    for v in t['sadp_verdicts']:
        if v['run']=='w1e_class_dependence':continue
        config=v['stored']['production_config']
        qs=[q for q in t['sadp_quality'] if q['run']==v['run'] and q['cell']==v['cell'] and q['config']==config]
        nsn=next(q for q in qs if q['recipe']=='nsn');plain=next(q for q in qs if q['recipe']=='kivi')
        gap=v['stored']['gap_at_prod']
        rows.append([v['cell'],config,str(nsn['k_bits']),pct(nsn['stats']['mean']),pct(plain['stats']['mean']),pct(gap['mean']),ci(gap)])
    table(['モデル＠長さ','設定','bit','NSN %','plain %','差 pp','差の95%区間'],rows)
    text('短区間4モデルでは、原記録で採用した設定において差の区間が0を超えたのは2モデル、0を含んだのは2モデル。OLMoEのA3 plainは30/30窓がNaNのため、2026-06-24に記録されたA2 fallbackを表示した。NaNセル自体も削除せず保持する。A3を実装できない4構成は「未実行」として別表に残す。')
    text('## T5. SADP：長区間の別モデル群（A0、INT5）')
    rows=[]
    for g in t['sadp_paired']:
        if g['run']!='w1e_class_dependence':continue
        qs=[q for q in t['sadp_quality'] if q['run']==g['run'] and q['cell']==g['cell']]
        a=next(q for q in qs if q['recipe']=='nsn');b=next(q for q in qs if q['recipe']=='kivi')
        rows.append([g['cell'],pct(a['stats']['mean']),pct(b['stats']['mean']),pct(g['stats']['mean']),ci(g['stats'])])
    table(['モデル＠長さ','NSN %','plain %','plain−NSN pp','95%区間'],rows)
    text('Qwen3.6では差の区間が0を含む。Qwen3との違いは調べたモデル・設定における記述であり、Kの局在性だけの因果効果として扱わない。')
    text('## T6. SADP：INT5短区間の点推定と区間')
    rows=[]
    for x in t['sadp_quality']:
        if x['run']=='e5_holdout_oracle' and x['recipe'] in ('nsn_int5','plain_asym_int5'):
            rows.append([x['cell'],x['recipe'],pct(x['stats']['mean']),ci(x['stats']),'はい' if x['point_within_plusminus_0_5'] else 'いいえ','はい' if x['ci_within_plusminus_0_5'] else 'いいえ'])
    table(['モデル','方式','平均 %','95%区間','点が±0.5以内','区間全体が±0.5以内'],rows)
    text('この表の両側帯判定は今回の説明用再計算。元の長区間コードのin_band（区間上限＜＋0.5%）と区別する。E5のthree-tier mask比較も機械表に保持したが、統合稿の一様NSN／plain比較とは分ける。')
    text('## T7. 実packingと品質の対応')
    table(['モデル','模擬QDQ PPL','実packed PPL','実packed−模擬','絶対差'],[[x['model'],f"{x['simulator_ppl']:.9f}",f"{x['packed_ppl']:.9f}",f"{x['packed_minus_simulator']:+.9f}",f"{x['absolute_difference']:.9f}"] for x in t['packed_ppl']])
    text('旧NSNのINT3非対称packingは、2048×128の合成tensorに補助情報を含めて3.140625 bit/要素。これをモデルのhead・層数へ拡大すると、FP16比80.371%減。モデル全体の実VRAM測定ではない。模擬QDQと実保存の合成比較では223,471/262,144要素がbit不一致、最大絶対差0.005859375。整数packingの可逆性と復元値の一致を分ける。')
    text('## T8. 旧NSNの逐次処理時間')
    table(['モデル','経路','ms/token（3反復の中央値）','非圧縮比 %'],[[x['model'],x['method'],f"{x['median_ms_per_token']:.2f}",pct(x['overhead_percent'])] for x in t['legacy_latency']])
    text('A100、prefix 1024、50逐次stepを計時。通常5step、compile経路は10stepの事前warmupが計時外で、厳密には計時開始時のcache長も異なる。roundtripは毎stepの再packing・復元。ほかの圧縮経路は初期packingが計時外。保存された小数2桁の時間から比を再計算しているため、旧要約の丸めた比とわずかに異なる。')
    text('## T9. SADPのQK score部品')
    table(['系列','モデル形状','有効／予定セル','NSN5がFP16より短い','FP16/NSN5 比中央値','code dtype'],[[x['run'],x['model_geometry'],f"{x['measured_cells']}/{x['planned_cells']}",f"{x['NSN5_shorter']}/{x['NSN5_vs_FP16_pairs']}",f"{x['FP16_over_NSN5_median']:.6f}",x['actual_codes_dtype']] for x in t['kernel_summary']])
    text('入力は合成K、終点はQK score。NSN5との対応比較は同一形状のFP16経路を対照とする。K1cの5欠測はGemma形状batch64・ctx65536の全5方式のOOM。uint8は1 code/byteであり、INT5 bitstreamの実packingではない。時間比＞1はNSN5が短い。')
    text('## T10. 再帰型二モデルの固定文章での品質')
    table(['モデル','PG19 book ID','予測数／文章','技術反復','融合の相対PPL %'],[[x['model'],x['book_id'],x['n_predictions'],x['technical_replicates'],pct(x['fused_change_percent'])] for x in t['loop_quality']])
    text(f"各文章を一度ずつNLL差で等重み集約した相対PPLはOuro {s['loop_aggregate_percent']['Ouro']:+.3f}%、Nanbeige {s['loop_aggregate_percent']['Nanbeige']:+.3f}%。技術反復を文章数へ足さない。5%は既存の診断目安であり、統計的非劣性境界として検定していない。")
    text('## T11. 再帰型二モデルの実保存量と時間')
    table(['モデル','非圧縮KV byte','圧縮KV byte','終端KV削減率 %'],[[x['model'],f"{x['baseline_bytes']:,}",f"{x['packed_bytes']:,}",f"{x['reduction_percent']:.3f}"] for x in t['loop_memory']])
    table(['モデル','計時範囲','対照','融合／対照','時間変化 %'],[[x['model'],x['scope'],x['reference'],f"{x['ratio']:.6f}",pct(x['change_percent'])] for x in t['loop_timing']])
    text('NATIVEも入力検査付き。OuroのPACKED_GROUPEDはK内部復元＋V全量復元、NanbeigeのPACKED_SDPAはK/V全量復元＋SDPA。二つの短縮率は同じ介入のモデル別効果量ではない。phase_sumはprefill＋初期圧縮＋8逐次step。終端KVはモデル全体のallocated／reserved peakと別の量である。')
    text('## T12. 同算術の橋渡し（VF01）')
    table(['候補','対照','融合／対照','時間変化 %'],[[x['candidate'],x['reference'],f"{x['ratio']:.6f}",pct(x['change_percent'])] for x in t['bridge']])
    text('2文章から6主試料。同算術で全量Vを作るTILE_MATERIALIZEとの比較は28.280%短縮。RF01対照は復元算術も異なる。原540計時から、各query・各方式の5反復中央値→対応比→非zeroの3query中央値→6試料中央値の順で再計算した。6独立文章と扱わない。原試験のH1不成立とH2/H3成立は維持する。')
    text('## 補助資料')
    text('88品質セル・54対応差・480部品セルと欠測を含む全行は抽出JSON、2,640窓の保存値は[window-values.json](data/extraction-v0.1/window-values.json)、重複窓数は抽出JSONのwindow_overlap。旧ノルム分散の8モデル相関は探索的補助証拠に置く（VのSpearman ρ＝0.881、K＝0.286）。')
    output.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data_dir,a.output)
