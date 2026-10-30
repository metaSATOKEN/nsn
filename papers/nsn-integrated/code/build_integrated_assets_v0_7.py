"""Build the integrated NSN Japanese draft from preserved extraction tables.
No model loading. Matplotlib layout follows the loop-kv paper asset generator.
Inputs remain read-only; --output-dir must be separate from experimental runs.
"""
import argparse, ast, hashlib, json, math, shutil, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from source_layout import source_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

DATE='2026-10-05'  # Existing observations; newly displayed INT5 rows.
MANUSCRIPT_DATE='2026-10-05'
TEAL='#087f8c'; ORANGE='#cf6d29'; INK='#213547'; GRAY='#64748b'

def table(headers, rows):
    def cell(x): return str(x).replace('|',r'\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |', '| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(cell,row))+' |' for row in rows])
def pct(x): return f'{x:+.3f}'
def short(model): return model.replace('gemma-4-E4B','Gemma-4-E4B').replace('Qwen3-30B-A3B','Qwen3-30B').replace('Gemma4-26B-A4B-base','Gemma4-26B').replace('Granite-3.0-3b-a800m','Granite-3B')
def main(root,out,template):
    out.mkdir(parents=True,exist_ok=True); figs=out/'figures/v0.2';figs.mkdir(parents=True,exist_ok=True)
    sources=[]
    def read(rel):
        p=source_file(root,rel);raw=p.read_bytes();sources.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest()});return json.loads(raw)
    t=read('papers/nsn-integrated/data/extraction-v0.1/evidence-tables.json')
    s=read('papers/nsn-integrated/data/extraction-v0.1/summary.json')
    lp=read('papers/loop-kv/paper-data-v0.2-2026-10-03.json')
    kv=read('runs/loop-kv-kv01-20260930-110929-fe60928a/offline-verification.json')
    gemma=[x for x in t['legacy_boundaries'] if x['run']=='phase21_gemma_b_anomaly_probe']
    gemma_rows=[]
    for label in ['SEG1','SEG2']:
        rows={x['method']:x for x in gemma if x['label']==label}
        base=rows['baseline']
        gemma_rows.append([label,base['tokens']//512,f"{base['ppl']:.4f}"]+[pct(rows[m]['percent']) for m in ['B-K-only','B-V-only','B-KV']])
    models=list(dict.fromkeys(x['model'] for x in t['legacy_ablation']))
    ablation=[[m]+[next(x for x in t['legacy_ablation'] if x['model']==m and x['method']==a)['percent'] for a in 'ABCD'] for m in models]
    boundary=[x for x in t['legacy_boundaries'] if 'D' in x['method'] and (x['run']!='phase20_bf16_disambiguation' or x['label']=='gemma-4-E4B/bf16')]
    boundary_rows=[]
    for x in boundary:
        descr=x['label'].replace('gemma-4-E4B','Gemma-4-E4B')
        if descr in ('SEG1','SEG2'): descr='Gemma-4-E4B 別区間'+descr[-1]
        match=re.search(r'INT(\d+)',x['method']); bits=int(match[1]) if match else 3
        n=f"{x['tokens']//512}窓／{x['tokens']:,}予測" if isinstance(x['tokens'],int) else '最大10窓・取得数未記載'
        boundary_rows.append([descr,f'非対称NSN {bits}bit',pct(x['percent']),n])
    prod=[]
    for v in t['sadp_verdicts']:
        if v['run']=='w1e_class_dependence':continue
        c=v['stored']['production_config'];qs=[q for q in t['sadp_quality'] if q['run']==v['run'] and q['cell']==v['cell'] and q['config']==c]
        nsn=next(q for q in qs if q['recipe']=='nsn');plain=next(q for q in qs if q['recipe']=='kivi')
        prod.append({'cell':v['cell'],'run':v['run'],'config':c,'bit':nsn['k_bits'],'nsn':nsn['stats']['mean'],'plain':plain['stats']['mean'],'gap':v['stored']['gap_at_prod'],'origins':[v['origin'],nsn['origin'],plain['origin']]})
    long=[g for g in t['sadp_paired'] if g['run']=='w1e_class_dependence']
    arch={'Bamba-9B-v2':'SSM＋attention','OLMoE-1B-7B':'MoE','Granite-3.0-3b-a800m':'MoE','Qwen3-30B-A3B':'MoE'}
    int5=[]
    original_path='Substrate-Aware-Diagnostic-and-Prescription/results/e5_holdout_oracle.json'
    original=read(original_path)
    for model,architecture in arch.items():
        q=next(q for q in t['sadp_quality'] if q['run']=='e5_holdout_oracle' and q['cell']==model and q['recipe']=='nsn_int5')
        raw=original['models'][model]['cells']['nsn_int5']
        vals=raw['per_window_dppl']
        assert len(vals)==30 and q['context']==1024 and q['k_bits']==q['v_bits']==5
        assert abs(sum(vals)/len(vals)-q['stats']['mean'])<1e-12
        assert all(abs(raw[k]-q['stats'][k])<1e-12 for k in ('mean','std','ci_lo','ci_hi'))
        assert q['origin']['sha256']==hashlib.sha256(source_file(root,original_path).read_bytes()).hexdigest()
        int5.append(dict(q,architecture=architecture))
    bitwidth_comparison=[]
    for model,architecture in arch.items():
        for bits in (3,4,5):
            q=next(q for q in t['sadp_quality'] if q['run']=='e5_holdout_oracle' and q['cell']==model and q['recipe']==f'nsn_int{bits}')
            raw=original['models'][model]['cells'][f'nsn_int{bits}']
            vals=raw['per_window_dppl']
            assert len(vals)==30 and q['context']==1024 and q['k_bits']==q['v_bits']==bits
            assert abs(sum(vals)/len(vals)-q['stats']['mean'])<1e-12
            assert all(abs(raw[k]-q['stats'][k])<1e-12 for k in ('mean','std','ci_lo','ci_hi'))
            assert q['origin']['sha256']==hashlib.sha256(source_file(root,original_path).read_bytes()).hexdigest()
            bitwidth_comparison.append(dict(q,architecture=architecture))
    int5_rows=[]
    for q in int5:
        cells=[next(x for x in bitwidth_comparison if x['cell']==q['cell'] and x['k_bits']==bits) for bits in (3,4,5)]
        int5_rows.append([q['cell'],q['architecture']]+[pct(x['stats']['mean']) for x in cells]+[f"[{pct(q['stats']['ci_lo'])}, {pct(q['stats']['ci_hi'])}]"])

    configs={'A2_causal_residual':'128区画＋高精度末尾','A3_post_norm':'128区画・末尾＋K正規化後'}
    sadp_rows=[[short(x['cell']),x['bit'],configs[x['config']],pct(x['nsn']),pct(x['plain']),pct(x['gap']['mean']),f"[{pct(x['gap']['ci_lo'])}, {pct(x['gap']['ci_hi'])}]"] for x in prod]
    latency_methods=['baseline','unpack_only','unpack_batched','unpack_compiled']
    latency_rows=[]
    for m in ['Qwen2-7B','Llama-3.1-8B']:
        latency_rows.append([m]+[f"{next(x for x in t['legacy_latency'] if x['model']==m and x['method']==a)['median_ms_per_token']:.2f}" for a in latency_methods])
    time_by={(x['model'],x['scope'],x['reference']):x for x in t['loop_timing']}
    model_rows=[]
    for mem in t['loop_memory']:
        m=mem['model'];q=next(x for x in t['loop_quality'] if x['model']==m)
        native=time_by[(m,'decode_240_positions' if m=='Ouro' else 'decode','NATIVE')]
        compressed=time_by[(m,'decode_240_positions' if m=='Ouro' else 'decode','PACKED_GROUPED' if m=='Ouro' else 'PACKED_SDPA')]
        model_rows.append([m,f"4×{q['n_predictions']}",pct(s['loop_aggregate_percent'][m]),f"{mem['baseline_bytes']:,} → {mem['packed_bytes']:,}",f"{mem['reduction_percent']:.3f}%",pct(compressed['change_percent']),pct(native['change_percent'])])
    texts=[]
    for x in t['loop_quality']:
        base=x['ppl']['NATIVE'];fused=x['ppl']['PACKED_KV_FUSED' if x['model']=='Ouro' else 'GQA_FUSED']
        texts.append([x['model'],x['book_id'],x['n_predictions'],f'{base:.6f}',f'{fused:.6f}',pct(x['fused_change_percent'])])
    snippets={
      'TABLE_NORM':table(['モデル','Kノルムの変動係数','Vノルムの変動係数','Bの相対PPL %'],[[x['model'],f"{x['k_cv_mean']:.6g}",f"{x['v_cv_mean']:.6g}",pct(x['B_percent'])] for x in t['norm_dispersion']]),
      'TABLE_KV_GEMMA':table(['区間','実取得窓数','非圧縮PPL','Kのみ %','Vのみ %','K/V両側 %'],gemma_rows),
      'TABLE_KV_OURO':table(['対象','相対PPL %','平均KL'],[[x['target'],pct(x['ppl_change_percent']),'—' if x['mean_kl'] is None else f"{x['mean_kl']:.9f}"] for x in kv['conditions']]),
      'TABLE_ABLATION':table(['モデル','A：行別対称','B：チャネル別非対称','C：対称NSN','D：非対称NSN'],[[r[0]]+[pct(v) for v in r[1:]] for r in ablation]+[['＋5%未満', '3/8','3/8','5/8','8/8']]),
      'TABLE_BOUNDARIES':table(['対象・条件','方式','相対PPL %','採点単位'],boundary_rows),
      'TABLE_INT5':table(['モデル','構成','3bit 平均 %','4bit 平均 %','5bit 平均 %','5bit 窓bootstrap 95%区間 %'],int5_rows),
      'TABLE_SADP':table(['モデル＠長さ','bit','設定','非対称NSN %','plain（ノルム分離なし）%','plain − NSN 差 pp','差の窓bootstrap 95%区間'],sadp_rows),
      'TABLE_LATENCY':table(['モデル','非圧縮','head別復元','head一括復元','compiled復元'],latency_rows),
      'TABLE_MODEL':table(['測定量','Ouro','Nanbeige'],[[lab,model_rows[0][i],model_rows[1][i]] for i,lab in enumerate(['モデル','文章×予測','融合の相対PPL %','非圧縮→圧縮KV byte','終端KV削減率','圧縮対照比の逐次処理時間 %','非圧縮比の逐次処理時間 %']) if i>0]),
      'TABLE_TEXTS':table(['モデル','PG19 ID','予測数','非圧縮PPL','融合PPL','相対変化 %'],texts),
    }
    core_path=Path(__file__).with_name('nsn_int3_core.py');core=core_path.read_text();tree=ast.parse(core)
    imports=[ast.get_source_segment(core,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));snippet='\n'.join(imports)+'\n\n'+ast.get_source_segment(core,fn)
    snippets['CORE_CODE']=snippet
    text=template.read_text()
    for key,value in snippets.items():text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    (out/f'manuscript-ja-v0.7-{MANUSCRIPT_DATE}.md').write_text(text)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':INK,'svg.fonttype':'none','svg.hashsalt':'nsn-integrated-v0.6','savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(figs/(name+'.svg'),bbox_inches='tight',metadata={'Date':DATE});fig.savefig(figs/(name+'.png'),dpi=175,bbox_inches='tight');plt.close(fig)
    # Figure 1: illustrative vector decomposition, not an empirical plot.
    fig=plt.figure(figsize=(12,6));ax=fig.add_axes([.05,.38,.32,.52]);ax.set_xlim(-2,34);ax.set_ylim(-2,44)
    for end,color,label in [((30,40),TEAL,'(30, 40): norm = 50'),((3,4),ORANGE,'(3, 4): norm = 5')]:
        ax.annotate('',xy=end,xytext=(0,0),arrowprops={'arrowstyle':'->','lw':3,'color':color})
    ax.text(11,37,'Same direction\n(0.6, 0.8)',fontsize=13,color=TEAL)
    ax.text(8,8,'norm = 5',color=ORANGE);ax.text(23,25,'norm = 50',color=TEAL)
    ax.set_xlabel('Coordinate 1');ax.set_ylabel('Coordinate 2');ax.set_title('Separate scale from direction',loc='left',fontweight='bold');ax.grid(alpha=.15)
    flow=fig.add_axes([.43,.15,.54,.77]);flow.axis('off')
    boxes=[(.04,.76,.92,.14,'Input KV block: T tokens × D channels',INK),(.04,.55,.39,.14,'Retain token scale\nStored norm (FP16 here)',TEAL),(.55,.55,.41,.14,'Quantize unit direction\nb-bit, per-channel min/max',ORANGE),(.04,.30,.92,.14,'Reconstruct direction; restore unit length',INK),(.04,.07,.92,.14,'Multiply by the saved norm → attention',TEAL)]
    for x,y,w,h,lab,c in boxes:
        flow.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',edgecolor=c,facecolor=c+'0c',linewidth=1.4));flow.text(x+w/2,y+h/2,lab,ha='center',va='center',fontsize=10)
    for start,end in [((.5,.75),(.23,.70)),((.5,.75),(.75,.70)),((.75,.54),(.75,.45)),((.5,.29),(.5,.22))]:flow.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':GRAY})
    flow.plot([.04,-.01,-.01],[.62,.62,.14],color=TEAL,lw=1.2,clip_on=False)
    flow.annotate('',xy=(.04,.14),xytext=(-.01,.14),arrowprops={'arrowstyle':'->','color':TEAL})
    flow.set_xlim(0,1);flow.set_ylim(0,1)
    fig.text(.05,.15,'Illustrative vectors.\nMetadata: norms, minima, and step sizes.',fontsize=10,color=GRAY)
    save(fig,'figure-01-nsn')
    # Figure 2: all four configurations, plus the preserved boundary rows.
    from matplotlib.colors import ListedColormap
    fig,axs=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[1.05,1]})
    values=np.array([r[1:] for r in ablation]);passed=values<5
    axs[0].imshow((~passed).astype(int),cmap=ListedColormap(['#d8efed','#f8dbc6']),vmin=0,vmax=1,aspect='auto')
    for i in range(len(ablation)):
        for j in range(4):axs[0].text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=10,color=INK)
    axs[0].set_yticks(range(len(ablation)),[short(r[0]) for r in ablation])
    axs[0].set_xticks(range(4),['A: row-sym.\n3/8','B: channel-asym.\n3/8','C: symmetric NSN\n5/8','D: asymmetric NSN\n8/8'],fontsize=8)
    axs[0].tick_params(axis='both',length=0)
    axs[0].set_title('A. Four configurations / same nominal 3-bit',loc='left',fontsize=11,pad=13)
    axs[0].set_xlabel('Relative PPL change (%)\nTeal: below +5%; orange: +5% or above',labelpad=12)
    chosen=[]
    for x in boundary:
        lab=x['label'].replace('gemma-4-E4B','Gemma-E4B').replace('/bf16',' BF16')
        if lab in ('SEG1','SEG2'): lab='Gemma-E4B segment '+lab[-1]
        match=re.search(r'INT(\d+)',x['method']); bits=int(match[1]) if match else 3
        lab += f' / {bits}-bit';chosen.append((lab,x['percent']))
    axs[1].barh(np.arange(len(chosen)),[v for _,v in chosen],color=[ORANGE if v>=5 else TEAL for _,v in chosen])
    axs[1].set_yticks(np.arange(len(chosen)),[a for a,_ in chosen],fontsize=8);axs[1].invert_yaxis()
    axs[1].set_title('B. Asymmetric NSN / changed conditions',loc='left',fontsize=11,pad=13)
    axs[1].axvline(5,ls='--',lw=1,color=ORANGE);axs[1].set_xlabel('Relative PPL change (%)');axs[1].xaxis.grid(alpha=.2);axs[1].set_axisbelow(True)
    fig.tight_layout(w_pad=4);save(fig,'figure-02-quality-boundaries')
    # Figure 3: INT5 applicability. Original pointwise window-bootstrap intervals.
    fig,ax=plt.subplots(figsize=(9.4,4.5))
    labels=[]
    for i,q in enumerate(int5):
        v=q['stats'];ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci_lo']],[v['ci_hi']-v['mean']]],fmt='o',color=TEAL,capsize=5,markersize=7,lw=2)
        labels.append(q['cell']+('\nSSM + attention' if i==0 else '\nMoE'))
        ax.text(.78,i,f"{v['mean']:+.3f}%",ha='right',va='center',fontsize=11,color=INK)
    ax.set_yticks(range(4),labels,fontsize=10);ax.invert_yaxis();ax.set_ylim(3.6,-.8);ax.set_xlim(-.45,.85)
    ax.axvline(0,color=GRAY,ls='--',lw=1);ax.xaxis.grid(alpha=.2)
    ax.set_xlabel('Relative PPL change (%) / mean and 95% window-bootstrap interval')
    ax.set_title('Asymmetric NSN at 5 bits across model architectures',loc='left',fontweight='bold',pad=30)
    ax.text(0,1.035,'Attention K/V only | full-window QDQ | 30 windows × 1,024 tokens per model',transform=ax.transAxes,fontsize=9,color=GRAY)
    fig.tight_layout();save(fig,'figure-03-int5-applicability')
    # Figure 4: original paired window bootstrap; no synthesized population CI.
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1,1]})
    groups=[[(x['cell'],x['gap']) for x in prod[:4]],[(x['cell'],x['gap']) for x in prod[4:]],[(x['cell'],x['stats']) for x in long]]
    titles=['A. Short context / 4-bit','B. Gemma long context / 5-bit','C. Qwen full-window / 5-bit']
    for ax,g,title in zip(axs,groups,titles):
        for i,(cell,v) in enumerate(g):
            ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci_lo']],[v['ci_hi']-v['mean']]],fmt='o',color=TEAL if v['ci_lo']>0 else GRAY,capsize=4)
        ax.set_yticks(range(len(g)),[short(c).replace('@','\nL=') for c,v in g],fontsize=9);ax.invert_yaxis();ax.set_ylim(len(g)-.5,-.5);ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set_title(title,loc='left',fontsize=11);ax.set_xlabel('Control without norm separation\n− asymmetric NSN (percentage points)',fontsize=9);ax.xaxis.grid(alpha=.2)
    fig.tight_layout(w_pad=2.6);save(fig,'figure-04-family-gaps')
    # Figure 5: each row has its own stated denominator.
    order=[('Ouro','decode_240_positions','NATIVE'),('Ouro','decode_240_positions','PACKED_GROUPED'),('Nanbeige','decode','NATIVE'),('Nanbeige','decode','PACKED_SDPA'),('Nanbeige','phase_sum','NATIVE')]
    labels=['Ouro decoding / checked uncompressed','Ouro decoding / K fused + full V','Nanbeige decoding / checked uncompressed','Nanbeige decoding / full KV + SDPA','Nanbeige with initial costs / checked uncompressed']
    ratios=[time_by[k]['ratio'] for k in order]
    fig,ax=plt.subplots(figsize=(11,4.7));y=np.arange(5);ax.barh(y,ratios,color=[TEAL if r<1 else ORANGE for r in ratios]);ax.set_yticks(y,labels,fontsize=10);ax.invert_yaxis();ax.axvline(1,color=INK,ls='--',lw=1);ax.set_xlim(0,1.30);ax.set_xlabel('Fused time / stated reference time');ax.set_title('Fused routes: decoding and initialization costs',loc='left',fontweight='bold');ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
    for i,r in enumerate(ratios):ax.text(r+.018,i,f'{r:.3f}',va='center',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    fig.tight_layout();save(fig,'figure-05-model-cost')
    data={'version':'0.4','date':DATE,'scope':'descriptive synthesis of existing evidence; no new model runs','source_files':sources,'norm_dispersion':t['norm_dispersion'],'kv_split_gemma':gemma,'kv_split_ouro':{k:kv[k] for k in ('conditions','comparisons','D','I','per_book','scope')},'int5_applicability':int5,'architecture_bitwidth_comparison':bitwidth_comparison,'ablation':t['legacy_ablation'],'boundaries':boundary,'adopted_conditions':prod,'long_context_gaps':long,'latency':t['legacy_latency'],'quality':t['loop_quality'],'timing':t['loop_timing'],'storage':t['loop_memory'],'bridge':t['bridge'],'summary':s,'inherited_loop_diagnostics':{k:v for k,v in lp.items() if k in ('input_validation','numerical_audit')},'generated_tables':snippets,'plot_rows':{'figure2_ablation':ablation,'figure2_boundaries':chosen,'figure3_int5':int5,'figure4_families':groups,'figure5_cost':[time_by[k] for k in order]}}
    (out/f'paper-data-v0.4-{DATE}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'manuscript_bytes':len(text.encode()),'figures':5,'generated_data_tables':10,'sources':sources},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--template',type=Path,default=Path(__file__).with_name('manuscript-ja-template-v0.7.md'));a=p.parse_args();main(a.research_root,a.output_dir,a.template)
