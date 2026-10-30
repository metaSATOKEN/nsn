"""Read-only extraction of previously verified results. No model/GPU/network use.
Usage: python build_paper_assets_v0_2.py --research-root PATH [--output-dir PATH]
The output directory must be a paper folder; raw evidence is never modified.
"""
import argparse, hashlib, json, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ap=argparse.ArgumentParser();ap.add_argument('--research-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
r=args.research_root.resolve();out=args.output_dir.resolve();(out/'figures/v0.2').mkdir(parents=True,exist_ok=True)
sources={}
def read(rel):
 p=r/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
a=read('reports/ac08-descriptive-statistics.json')
b=read('throughline/audits/loop-kv-nb05-results-v0.1-2026-10-03/supplementary-analysis.json')
iv=read('reports/iv01-descriptive-statistics.json')
v=read('throughline/audits/loop-kv-vf01-results-v0.1-2026-10-02/analysis.json')
ac=read('plans/loop-kv-ac08-input-contract-v0.1-2026-10-02.json')
nb=read('plans/loop-kv-nb05-input-contract-v0.1-2026-10-03.json')
# Check claimed dimensions against frozen implementation / saved model config.
src=r/'scripts/loop_kv_ac08.py';sources[str(src.relative_to(r))]=hashlib.sha256(src.read_bytes()).hexdigest()
assert '(16,16,24)' in src.read_text()
nconf=read('runs/loop-kv-nb05-20261003-090604-18baabf5/loaded_config.json')
assert nconf['num_attention_heads']==48 and nconf['num_key_value_heads']==8
assert len(a['quality'])==4 and len(b['books'])==4 and len(a['comparisons'])==8
assert len(a['repeats'])==12 and all(x['logits']['bit_differences']==0 for x in a['repeats'])
assert a['hypotheses']['H4_same_packed_storage'] and all(b['hypotheses'].values())
assert all(v['route_counts'][k]['pass_count']==v['route_counts'][k]['total'] for k in ['bridge_v_equal','bridge_context_equal','bridge_output_equal'])
assert v['hypotheses']=={'H1':False,'H2':True,'H3':True}
# Header + norm/channel metadata included in real packed storage.
block_bytes=64*128*3//8+(64+128+128)*2+12
assert block_bytes==3724
ouro_expected=192*16*2*((896//64)*block_bytes+127*128*2)
nb_expected=44*8*2*(block_bytes+72*128*2)
assert ouro_expected==a['memory']['PACKED_KV_FUSED']['terminal_storage']==520077312
assert nb_expected==b['books'][0]['terminal_kv_bytes']['GQA_FUSED']==15597824
# Tables retain source aggregation, not ratios of separately aggregated medians.
quality=[]
for x in a['quality']:
 pair=next(z for z in x['pairs'] if z['candidate']=='PACKED_KV_FUSED' and z['reference']=='NATIVE')
 w=next(z for z in a['hypotheses']['H3_quality']['cases'] if z['book_id']==x['book_id'] and z['round']==0 and z['window']=='W4')
 quality.append(dict(model='Ouro',book_id=x['book_id'],ppl=x['ppl'],fused_change_percent=pair['ppl_change_pct'],late_window_change_percent=w['ppl_change_pct']))
for x in b['books']:
 q=x['quality']['post_compression_8'];quality.append(dict(model='Nanbeige',book_id=x['book_id'],ppl=q['ppl'],fused_change_percent=q['contrasts']['GQA_FUSED_vs_NATIVE']['relative_ppl_percent']))
# AC08 cross-book summary is descriptive, once per book (repeats identical).
ac_all=100*math.expm1(statistics.mean(next(z for z in x['pairs'] if z['candidate']=='PACKED_KV_FUSED' and z['reference']=='NATIVE')['delta_nll'] for x in a['quality']))
ac_late=100*math.expm1(statistics.mean(z['delta_nll'] for z in a['hypotheses']['H3_quality']['cases'] if z['round']==0 and z['window']=='W4'))
data={'source_sha256':sources,'scope':'Existing-results extraction; no new inference or hypothesis test.','block_storage_bytes':block_bytes,'quality':quality,'ouro':{'timing':a['timing'],'ratios':a['time_ratio_summary'],'memory':a['memory'],'aggregate_ppl_change_percent':{'ALL':ac_all,'W4':ac_late}},'nanbeige':{'timing':b['timing'],'books':[{'id':x['book_id'],'timing':x['timing'],'memory':x['memory'],'terminal_kv_bytes':x['terminal_kv_bytes']} for x in b['books']],'quality':b['quality']},'bridge':{'time':v['main_time_ms'],'ratios':{k:z['main'] for k,z in v['timing']['cuda_ms'].items()},'memory':v['main_memory_bytes'],'original_hypotheses':v['hypotheses']}}
data['input_validation']=iv
(out/'paper-data-v0.2-2026-10-03.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
# Figure 1: reconstructed values held constant in the bridge contrast.
f,ax=plt.subplots(figsize=(11,4));ax.set_xlim(0,11);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,.65,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#536475'))
 ax.text(x+w/2,y+.325,text,ha='center',va='center',fontsize=10)
for y,lab in [(2.8,'Existing reconstruction'),(1.55,'Bridge control'),(.3,'Fused reconstruction')]:ax.text(.05,y+.325,lab,ha='left',va='center',weight='bold')
for y in [2.8,1.55,.3]:box(2.6,y,1.4,'Same packed V','#e6edf5')
box(4.55,2.8,2.5,'Existing restore\nfull V in GPU memory','#fff0dd');box(7.7,2.8,2.9,'Attention + checks','#e2efe7')
box(4.55,1.55,2.5,'Tile restore\nfull V in GPU memory','#e4e8ff');box(7.7,1.55,2.9,'Attention + checks','#e2efe7')
box(4.55,.3,6.05,'Tile restore inside attention + checks\nNo full-length restored V buffer','#d5eee4')
for y in [2.8,1.55,.3]:ax.annotate('',xy=(4.45,y+.325),xytext=(4.08,y+.325),arrowprops=dict(arrowstyle='->',color='#536475'))
for y in [2.8,1.55]:ax.annotate('',xy=(7.58,y+.325),xytext=(7.14,y+.325),arrowprops=dict(arrowstyle='->',color='#536475'))
ax.text(7.5,1.18,'Bridge and fused: same restored values on tested inputs',ha='center',fontsize=9,color='#285941')
f.tight_layout();f.savefig(out/'figures/v0.2/figure-01-bridge.svg');f.savefig(out/'figures/v0.2/figure-01-bridge.png',dpi=180);plt.close(f)

def save(fig,name):
 for ext in ['svg','png']:
  fig.savefig(out/'figures/v0.2'/f'{name}.{ext}',dpi=180)
 plt.close(fig)

# Figure 2: each memory metric normalized to its own checked-native reference.
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharex=True)
labels=['Terminal KV storage','Decode allocated peak','Recorded allocated peak\n(including prefill)','Decode reserved peak']
mem_plot={}
for ax,model in zip(axes,['Ouro','Nanbeige']):
 if model=='Ouro':
  arms=['NATIVE','PACKED_GROUPED','PACKED_KV_FUSED']
  matrix=[[a['memory'][arm]['terminal_storage'],a['memory'][arm]['allocated_peak'],a['memory'][arm]['recorded_prefill_inclusive_peak']['allocated_peak'],a['memory'][arm]['reserved_peak']] for arm in arms]
 else:
  arms=['NATIVE','PACKED_SDPA','GQA_FUSED'];book=b['books'][0]
  matrix=[[book['terminal_kv_bytes'][arm],book['memory'][arm]['allocated']['median_decode_mib']*2**20,book['memory'][arm]['allocated']['median_whole_main_mib']*2**20,book['memory'][arm]['reserved']['median_decode_mib']*2**20] for arm in arms]
 ratios=[[100*x/y for x,y in zip(row,matrix[0])] for row in matrix[1:]]
 mem_plot[model]={'labels':labels,'packed_percent':ratios[0],'fused_percent':ratios[1]}
 for off,values,color,label in [(-.17,ratios[0],'#738eae','Packed comparison'),(.17,ratios[1],'#217c66','K/V fused')]:
  bars=ax.barh([i+off for i in range(4)],values,height=.29,color=color,label=label)
  for bar,val in zip(bars,values):ax.text(val+1.3,bar.get_y()+bar.get_height()/2,f'{val:.2f}%',va='center',fontsize=8)
 ax.axvline(100,color='#444',ls='--',lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,122);ax.set_xlabel('Percent of checked native, for each metric');ax.set_title(model,weight='bold');ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',frameon=False)
fig.subplots_adjust(left=.22,right=.98,top=.88,bottom=.2,wspace=.86)
save(fig,'figure-02-memory')

# Figure 3: paired interpretation remains in text; bars are level medians, not CIs.
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
barlabels=['Existing\nfull V','Tile full V\n(bridge)','Tile V\nfused']
arms=['RF01_MATERIALIZE','TILE_MATERIALIZE','V_FUSED'];colors=['#9eabbc','#849bd1','#217c66']
vals_time=[v['main_time_ms'][arm]['cuda_ms'] for arm in arms]
vals_mem=[v['main_memory_bytes'][arm]['allocated_extra']['min']/2**20 for arm in arms]
for ax,vals,ylab,title in zip(axes,[vals_time,vals_mem],['CUDA time (ms / call)','Allocated increase (MiB)'],['Reconstruction + attention + checks','Temporary allocation above starting state']):
 bars=ax.bar(range(3),vals,color=colors,width=.63)
 for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+max(vals)*.04,f'{val:.4f}',ha='center',fontsize=10)
 ax.set_xticks(range(3),barlabels);ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel(ylab);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout(pad=1.7);save(fig,'figure-03-bridge-cost')

# Figure 4: keep quality alongside the corresponding results section.
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharex=True)
for ax,model,title in zip(axes,['Ouro','Nanbeige'],['Ouro: 255 predictions / text','Nanbeige: 8 predictions / text']):
 rows=[x for x in quality if x['model']==model];ys=list(range(4))
 ax.scatter([x['fused_change_percent'] for x in rows],ys,color='#227766',s=44,label='Main window')
 if model=='Ouro':ax.scatter([x['late_window_change_percent'] for x in rows],ys,marker='s',facecolors='none',edgecolors='#9f5b24',s=44,label='Last 64 predictions')
 ax.axvline(0,color='.55',lw=.8);ax.axvline(5,color='#ad5555',ls='--',lw=1)
 ax.set_yticks(ys,[x['book_id'] for x in rows]);ax.invert_yaxis();ax.set_xlim(-2.5,5.7);ax.set_xlabel('PPL change vs checked native (%)');ax.set_title(title,fontsize=11);ax.set_ylabel('PG19 book ID');ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.15)
fig.tight_layout(pad=1.5);save(fig,'figure-04-quality')

# Figure 5: keep each contrast's own denominator and preregistered aggregation.
fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1,1.3]})
time_plot={};colors=['#326894','#217c66','#326894','#217c66']
ouro_rows=[]
for reference,label in [('PACKED_GROUPED','Decode / K-fused, V-full'),('NATIVE','Decode / checked native')]:
 vals=[next(z for z in c['pairs'] if z['candidate']=='PACKED_KV_FUSED' and z['reference']==reference)['time_ratio'] for c in a['comparisons']]
 ouro_rows.append({'label':label,'ratios':vals,'median':statistics.median(vals),'unit':'book x technical repeat (8 blocks; 4 books)'})
nb_rows=[]
for scope,base,label in [('decode','PACKED_SDPA','Decode / full-restore SDPA'),('decode','NATIVE','Decode / checked native'),('phase_sum','PACKED_SDPA','Including initial cost /\nfull-restore SDPA'),('phase_sum','NATIVE','Including initial cost /\nchecked native')]:
 z=b['timing']['cuda_ms'][scope]['GQA_FUSED_vs_'+base]
 nb_rows.append({'label':label,'ratios':z['book_medians'],'median':z['ratio'],'unit':'book median (4 books; 3 technical repeats each)'})
for ax,rows,title,limits in zip(axes,[ouro_rows,nb_rows],['Ouro: decode interval','Nanbeige: decode and initial cost'],[(-10,10),(-49,23)]):
 for j,z in enumerate(rows):
  vals=[100*(x-1) for x in z['ratios']];offset=[(i-(len(vals)-1)/2)*(.16/max(len(vals)-1,1)) for i in range(len(vals))]
  ax.scatter(vals,[j+o for o in offset],s=31,color=colors[j],alpha=.85)
  med=100*(z['median']-1);ax.plot([med,med],[j-.19,j+.19],color='#1c2b30',lw=2)
  ax.annotate(f'{med:+.2f}%',(med,j),xytext=(4,-17),textcoords='offset points',fontsize=9)
 ax.axvline(0,color='.45',lw=1);ax.set_yticks(range(len(rows)),[z['label'] for z in rows]);ax.set_ylim(len(rows)-.5,-.5);ax.set_xlim(*limits);ax.set_xlabel('Fused time change vs stated comparator (%)');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.15)
fig.subplots_adjust(left=.19,right=.965,top=.86,bottom=.17,wspace=1.05)
save(fig,'figure-05-model-time')
time_plot['Ouro']=ouro_rows;time_plot['Nanbeige']=nb_rows
# Keep plotted values, units and level-vs-ratio distinctions machine-readable.
data['figure_metrics']={'figure_02_memory':mem_plot,'figure_03_component':{'arm_order':arms,'cuda_ms':vals_time,'allocated_extra_mib':vals_mem},'figure_05_time':time_plot}
(out/'paper-data-v0.2-2026-10-03.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'quality_rows':len(quality),'source_files':len(sources),'figures':5,'original_hypotheses_preserved':data['bridge']['original_hypotheses']},indent=2))
