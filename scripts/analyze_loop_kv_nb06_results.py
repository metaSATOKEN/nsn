"""Offline descriptive NB06 reaggregation from per-step records; no model imports.
Usage: python analyze_loop_kv_nb06_results.py RUN_DIRECTORY OUTPUT_REPORTS
The fixed verifier is a separate prerequisite, not replaced by this script.
"""
import argparse, hashlib, json, math, statistics as st
from pathlib import Path

ARMS = ['NATIVE', 'PACKED_SDPA', 'GQA_FUSED']
WINDOWS = {'full_256': (1, 256), 'first_8': (1, 8), 'block_1': (1, 64),
           'block_2': (65, 128), 'block_3': (129, 192), 'block_4': (193, 256)}
CHECKPOINTS = [8, 64, 128, 256]
BASE = 'nb06-results-v0.1-2026-10-08'

def read(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

def from_stats(x):
    s = x['nll_components']
    value = s['max_logit'] + math.log(s['sum_exp_shifted']) - s['target_logit']
    close(value, x['nll'])
    return value

def analyze(root):
    summary = read(root/'summary.json')
    books = []; nll_count = 0; errors = []
    for bi, original in enumerate(summary['books']):
        path = root/'books'/f'b{bi}'
        all_rows = {}
        for spec in rows(path/'trajectories.jsonl'):
            rr = rows(path/'series'/spec['id']/'steps.jsonl')
            assert [x['step'] for x in rr] == list(range(257))
            for x in rr:
                errors.append(abs(from_stats(x) - x['nll'])); nll_count += 1
            all_rows[spec['id']] = rr
        quality = {}
        for window, (lo, hi) in WINDOWS.items():
            nll = {a: st.median(st.mean(from_stats(x) for x in all_rows[f'main-r{rep}-{a.lower()}'][lo:hi+1])
                               for rep in range(3)) for a in ARMS}
            delta = {a: 100*math.expm1(nll[a]-nll['NATIVE']) for a in ARMS[1:]}
            for a in ARMS:
                close(nll[a], original['quality'][window]['median_mean_nll'][a])
            quality[window] = {'mean_nll': nll, 'ppl': {a: math.exp(v) for a, v in nll.items()},
                               'relative_ppl_percent_vs_native': delta}
        costs = {}
        for field in ['cuda_ms', 'wall_ms']:
            costs[field] = {}
            for end in CHECKPOINTS:
                arm_costs = {}
                for a in ARMS:
                    reps = []
                    for rep in range(3):
                        rr = all_rows[f'main-r{rep}-{a.lower()}']; measured = rr[1:end+1]
                        prefill = rr[0]['model_timing'][field]
                        initial = (rr[0]['compression_timing'] or {}).get(field, 0.)
                        forward = sum(x['model_timing'][field] for x in measured)
                        post = sum(x['post_timing'][field] for x in measured)
                        flush = sum(x['post_timing'][field] for x in measured if x['step'] in [64,128,192,256])
                        reps.append(dict(prefill_ms=prefill, initial_compression_ms=initial, forward_ms=forward,
                            post_decode_ms=post, flush_steps_post_ms=flush, other_post_ms=post-flush,
                            decode_ms=forward+post, phase_sum_ms=prefill+initial+forward+post))
                    arm_costs[a] = reps
                ratios = {}
                for metric in ['decode_ms', 'phase_sum_ms', 'forward_ms']:
                    ratios[metric] = {}
                    for ref in ['NATIVE','PACKED_SDPA']:
                        values = [arm_costs['GQA_FUSED'][i][metric]/arm_costs[ref][i][metric] for i in range(3)]
                        ratios[metric][ref] = dict(paired_ratios=values, median_ratio=st.median(values))
                        if metric != 'forward_ms':
                            close(st.median(values),original['timing'][field]['checkpoints'][str(end)]['ratios'][metric][ref]['median_ratio'])
                costs[field][str(end)] = dict(by_arm=arm_costs, ratios=ratios)
        memory = {}
        for a in ARMS:
            mem = []
            for rep in range(3):
                rr = all_rows[f'main-r{rep}-{a.lower()}']
                mm = [x[k] for x in rr for k in ['model_timing','compression_timing','post_timing'] if x[k]]
                dm = [x[k] for x in rr[1:] for k in ['model_timing','post_timing'] if x[k]]
                measured = dict(terminal_kv_bytes=rr[-1]['inventory']['unique_storage_bytes'],
                    full_peak_allocated=max(x['allocated_peak'] for x in mm),
                    full_peak_reserved=max(x['reserved_peak'] for x in mm),
                    decode_peak_allocated=max(x['allocated_peak'] for x in dm),
                    decode_peak_reserved=max(x['reserved_peak'] for x in dm))
                assert measured['terminal_kv_bytes'] == original['terminal_kv_bytes'][a]
                for k, v in measured.items():
                    ref = original['memory_peak_bytes'][a]['256'][rep]
                    target = ref['kv_storage_bytes'] if k == 'terminal_kv_bytes' else ref['full_phase_peak' if k.startswith('full') else 'decode_peak'][k.rsplit('_',1)[1]]
                    assert v == target
                mem.append(measured)
            memory[a] = mem
        books.append(dict(book_id=original['book_id'],quality=quality,costs=costs,memory=memory))
    aggregate_quality = {}
    for w in WINDOWS:
        means = {a: st.mean(b['quality'][w]['mean_nll'][a] for b in books) for a in ARMS}
        pct = {a:100*math.expm1(means[a]-means['NATIVE']) for a in ARMS[1:]}
        for a,v in pct.items():close(v,summary['quality'][w]['contrasts'][a+'_vs_NATIVE']['relative_ppl_percent'])
        aggregate_quality[w] = dict(mean_nll=means,ppl={a:math.exp(v) for a,v in means.items()},relative_ppl_percent_vs_native=pct)
    aggregate_timing = {}
    for field in ['cuda_ms','wall_ms']:
        aggregate_timing[field] = {}
        for end in CHECKPOINTS:
            out = {}
            for metric in ['decode_ms','phase_sum_ms','forward_ms']:
                out[metric] = {}
                for ref in ['NATIVE','PACKED_SDPA']:
                    vals = [b['costs'][field][str(end)]['ratios'][metric][ref]['median_ratio'] for b in books]
                    out[metric][ref] = st.median(vals)
                    if metric != 'forward_ms':close(st.median(vals),summary['timing'][field][str(end)][metric]['GQA_FUSED_vs_'+ref]['median_of_book_medians'])
            aggregate_timing[field][str(end)] = out
    mean_costs = {a:{k:st.mean(x[k] for b in books for x in b['costs']['cuda_ms']['256']['by_arm'][a])
                   for k in books[0]['costs']['cuda_ms']['256']['by_arm'][a][0]} for a in ARMS}
    model_ready=read(root/'model_ready.json')
    last=read(root/'books/b3/book_complete.json')
    return dict(run_id=root.name,summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        nll_components_checked=nll_count,max_nll_component_difference=max(errors),
        measurement_elapsed_minutes=(last['finished_monotonic']-model_ready['loaded_monotonic'])/60,
        quality=aggregate_quality,timing=aggregate_timing,mean_additive_cost_cuda_ms=mean_costs,
        storage_reduction_percent=100*(1-24642816/69206016),books=books,
        aggregation='Prespecified paired medians within each text, then median of four text medians. Additive cost table uses means solely for accounting.',
        new_model_calls=0)

def plot(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    for b,col in zip(data['books'],colors):
        ax[0].plot([1,2,3,4],[b['quality'][f'block_{i}']['relative_ppl_percent_vs_native']['GQA_FUSED'] for i in range(1,5)],'o-',color=col,label=b['book_id'])
    ax[0].axhline(5,color='#B22222',ls='--',lw=1,label='5% threshold')
    ax[0].axhline(0,color='grey',lw=.7)
    ax[0].set(title='A  Quality after repeated updates',ylabel='PPL change vs checked native (%)',xlabel='Decode steps (64-token windows)',xticks=[1,2,3,4],xticklabels=['1–64','65–128','129–192','193–256'])
    ax[0].legend(title='Text ID',fontsize=8,ncol=2,loc='upper left')
    for metric,label,col in [('decode_ms','Decode + cache updates','#0072B2'),('phase_sum_ms','Including prefill + initial packing','#D55E00')]:
        ax[1].plot(range(4),[100*(data['timing']['cuda_ms'][str(n)][metric]['NATIVE']-1) for n in CHECKPOINTS],'o-',label=label,color=col)
    ax[1].axhline(0,color='grey',ls='--',lw=1)
    ax[1].set(title='B  Cumulative cost of fused route',xlabel='Decode steps',ylabel='Time change vs checked native (%)',xticks=range(4),xticklabels=CHECKPOINTS)
    ax[1].legend(fontsize=8)
    ax[1].annotate('256 steps: +0.003% total\nApproximately equal',xy=(3,0),xytext=(1.3,5.5),arrowprops=dict(arrowstyle='->',color='#444444'),fontsize=9)
    for i,(b,col) in enumerate(zip(data['books'],colors)):
        vs=[100*(x-1) for x in b['costs']['cuda_ms']['256']['ratios']['phase_sum_ms']['NATIVE']['paired_ratios']]
        ax[2].scatter(vs,[i-.1,i,i+.1],color=col,s=28,alpha=.7)
        ax[2].scatter([st.median(vs)],[i],marker='|',color='black',s=250,zorder=4)
    ax[2].axvline(0,color='grey',ls='--',lw=1)
    ax[2].set(title='C  No consistent total-time advantage',xlabel='256-step total-time change (%)',yticks=range(4),yticklabels=[b['book_id'] for b in data['books']],ylabel='Text ID',xlim=(-.9,.5))
    ax[2].text(.03,.98,'Dots: 3 paired technical replicates\nBlack mark: within-text median',transform=ax[2].transAxes,va='top',fontsize=8)
    fig.suptitle('NB06 · Nanbeige · fixed 128-token prefix + 256 teacher-forced steps',fontsize=13)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.svg'));plt.close(fig)

def report(d, out):
    qualityrows=[]
    for b in d['books']:
        vals=[b['quality'][w]['relative_ppl_percent_vs_native']['GQA_FUSED'] for w in WINDOWS]
        qualityrows.append('| '+b['book_id']+' | '+' | '.join(f'{v:+.3f}%' for v in vals)+' |')
    times=[]
    for n in CHECKPOINTS:
        a=d['timing']['cuda_ms'][str(n)]
        times.append(f"| {n} | {100*(a['decode_ms']['NATIVE']-1):+.3f}% | {100*(a['phase_sum_ms']['NATIVE']-1):+.3f}% | {100*(a['decode_ms']['PACKED_SDPA']-1):+.3f}% |")
    costs=[]
    labels=dict(NATIVE='非圧縮（検査付き）',PACKED_SDPA='圧縮＋全量復元SDPA',GQA_FUSED='圧縮＋融合')
    for a,x in d['mean_additive_cost_cuda_ms'].items():
        costs.append('| '+labels[a]+' | '+' | '.join(f'{x[k]:.2f}' for k in ['prefill_ms','initial_compression_ms','forward_ms','post_decode_ms','phase_sum_ms'])+' |')
    mem=[]
    for a in ARMS:
        x=d['books'][0]['memory'][a][0]
        mem.append('| '+labels[a]+' | '+' | '.join(f'{x[k]/2**20:,.3f}' for k in ['terminal_kv_bytes','decode_peak_allocated','full_peak_allocated','full_peak_reserved'])+' |')
    pairs=[]
    for b in d['books']:
        rr=b['costs']['cuda_ms']['256']['ratios']['phase_sum_ms']['NATIVE']
        pairs.append(f"| {b['book_id']} | {100*(rr['median_ratio']-1):+.4f}% | {min(100*(v-1) for v in rr['paired_ratios']):+.4f}〜{max(100*(v-1) for v in rr['paired_ratios']):+.4f}% | {sum(v<1 for v in rr['paired_ratios'])}/3 |")
    text=f'''# NB06：圧縮更新を伴う256 step継続の結果

2026-10-08。run `{d['run_id']}`。**全60系列・15,420 forward完了、別実装のオフライン照合でもPASS_MODEL_CONNECTION。**

**継続区間を8から256 stepへ延ばし、途中の圧縮更新を繰り返しても、今回の4文章では品質の5%目安を満たした。終端KVは{d['storage_reduction_percent']:.3f}%減。融合は全量復元より速い一方、非圧縮に対しては初期・追加圧縮込みでほぼ同時間だった。**

| 主な結果 | 観測 |
|---|---|
| 融合の相対PPL・全256予測 | 4文章を一度ずつ集約して **+0.152%** |
| 各文章の全256 / 終盤64の最大悪化 | **+0.429% / +2.545%**（各4/4で5%以内） |
| 終端KV実storage | **66.000 → 23.501 MiB、{d['storage_reduction_percent']:.3f}%減** |
| 融合 / 全量復元・decode＋追加圧縮 | **42.500%短縮** |
| 融合 / 非圧縮・decode＋追加圧縮 | **0.604%短縮** |
| 融合 / 非圧縮・prefill＋初期圧縮＋decode＋追加圧縮 | **+0.003%**、実質的に同程度 |
| GPU allocatedの全工程peak | **8,197.590 MiBで3方式同じ** |

時間の主集約は、同じ文章・同じ反復の対応比を3反復で中央値にし、その4文章の中央値を取ったCUDA event値。比較相手は入力検査付きの実装。wallも同方向で、256 stepの総時間差は+0.004%。以下の費用分解のみ加算できる平均を使う。

![品質と累積時間](figures/{BASE}.png)

## 1. 固定条件と完了範囲

Nanbeige4.2-3B-Base、BF16、batch 1、44 cache slot（22層×2周）、Q48/KV8/head128。RTX PRO 6000 Blackwell Server Edition、torch2.11.0+cu128、Triton3.6.0、transformers4.45.1。同じ4冊・最初の128 tokenからteacher forcingで256 step、終端cache長384。計測フェーズの記録時間は約{d['measurement_elapsed_minutes']:.1f}分（環境準備・モデル取得・照合・ZIPを含まない）。

非圧縮、圧縮＋全量復元SDPA、圧縮＋GQA融合を各3技術反復。公式一致、診断、全長warmupを加えて60系列。64-token blockのNSN3と64 token以上のBF16末尾を保持し、step64/128/192/256の4回追加圧縮。最後の圧縮も費用と保存量に含み、その後のattention利用まで確認した更新は3回。既存blockの再量子化・材料の差替えはない。

## 2. 品質：全体でも終盤でも5%目安内

表は融合対非圧縮の相対PPL。負値は予測損失の低下を表す。

| PG19 ID | 全256 | 最初8 | 1–64 | 65–128 | 129–192 | 193–256 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(qualityrows)}

4文章集約PPLは非圧縮8.536475、全量復元8.551620、融合8.549434。融合の終盤64集約は+0.308%。全量復元の全256集約は+0.177%。両圧縮経路とも、全文章・全指定窓で5%以内だった。全256の平均が小さいことだけに依存せず、終盤を別に確認できた。

最初8予測の融合集約+1.526%と全256の+0.152%は評価するtoken集合が違う。「長くすると品質が改善する」という因果比較ではない。同方式3反復は全logit hashが一致した。

## 3. 時間：初期費用との差は縮まったが、優位とは判定しない

| 継続step | 融合/非圧縮：decode＋追加圧縮 | 融合/非圧縮：初期費用込み | 融合/全量復元：decode＋追加圧縮 |
|---|---:|---:|---:|
{chr(10).join(times)}

8 stepで+14.334%だった初期費用込みの差は、256 stepで+0.003%まで縮まった。取得前の「集約比が1未満」には固定4点のどこでも到達しなかった。一方、256 stepでは文章中央値が非圧縮より短いもの2冊・長いもの2冊、反復も短い6/12・長い6/12である。**この結果の読み方は「ほぼ同時間」であり、明確な低速化や高速化の実証ではない。**

| PG19 ID | 256 step総時間差の対応中央値 | 3技術反復の範囲 | 融合が短い反復 |
|---|---:|---:|---:|
{chr(10).join(pairs)}

### 保存記録による費用分解（記述的な事後分析）

単位ms。4文章×3反復の算術平均なので列を加算できる。上の主集約を置換しない。

| 方式 | prefill | 初期圧縮 | 256 forward | 256 post処理 | 合計 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(costs)}

融合のforward部分は非圧縮より平均約296.27 ms短い。その一方、4回の圧縮更新を含むpost処理は233.15 ms（非圧縮1.64 ms）、初期圧縮は54.71 ms。融合postのうち約230.99 ms、99.07%が4更新stepに集まる。**記録上、forward側の節約の大部分と釣り合う追加費用が圧縮更新・初期圧縮にある。** 個々の圧縮演算の因果寄与までは、この工程集計では分解していない。

平均内訳の合計では融合がわずかに短いが、主集約の対応中央値は+0.003%。集約方法を結果に合わせて取り替えず、どちらからも大きな総時間優位を読み取らない。次に速度を改善するなら、継続長だけを増やすより、追加blockの圧縮・連結費用を切り分ける根拠が得られた。

## 4. 保存量とGPU peak

単位MiB（2^20 byte）。全4文章・全3主反復で各値は同じ。

| 方式 | 終端KV実storage | decode allocated peak | 全工程 allocated peak | 全工程 reserved peak |
|---|---:|---:|---:|---:|
{chr(10).join(mem)}

圧縮経路の終端は24,642,816 byte。コード10,813,440、補助情報2,252,800、header42,240、BF16末尾11,534,336の和で、実storageと一致する。非圧縮69,206,016 byteとの差は42.499 MiB。

融合のdecode allocated peakは非圧縮より42.089 MiB低いが、全工程peakはprefillが支配して同じ。KVの64.4%削減をGPU全体の64.4%削減とは扱わず、保存量と各工程のpeakを分けて示す。

## 5. 照合と取得前予想

- ZIP全303ファイルのSHA256と、配布した29ソースの一致を確認。原ZIPは225,291,002 byte（約214.854 MiB）。
- 配布前固定のNumPy検証器で公式/NATIVE全位置logit・終端KV一致、経路、全step実storage、追加blockと既存block不変性、同方式反復、warmup後のcache不変を再照合。
- 固定32点の実Q数値検査は全点成立。融合と橋渡し対照のK/V復元は同一、工程別固定予算内。これは全位置の生KV保存とは異なる。
- 全15,420位置の保存損失成分を再計算。固定384行の全語彙logitsからの検算も既存検証器が実施。非保存位置の生logitsは復元できず、損失成分とhashを点検する範囲。
- ローカル再計算とColab集計に42個の微小な派生浮動小数差、最大3.553e-15。固定比較規則内で同一判定。原summaryを書き換えていない。

| 取得前予想 | 今回 |
|---|---|
| H1：4文章の接続・保存・固定数値・反復安定 | 成立 |
| H2：融合の全256/終盤64が全4文章で5%以内 | 成立 |
| H3：256 step融合decodeが全量復元より短い | 成立、42.500%短縮 |
| 非圧縮への総時間償却 | 方向予想なし。固定4点の集約では1未満なし、256でほぼ同時間 |

## 6. 到達点と次の扱い

今回加わった証拠は、**同じ材料で継続を延ばし、圧縮更新を繰り返した実経路でも、短区間の品質とKV削減が保たれた**こと。融合を使うと全量復元の費用を抑えられるが、非圧縮との総時間差は更新費用も含めて評価する必要があることも確認した。

範囲は4文章、128＋256 token、batch 1、teacher forcing、一つのGPU/実装。技術反復と部分窓を独立文章数へ加えず、旧NB05とのプール、p値・区間推定、未測定長への償却点外挿は行わない。長い自由生成や数千token文脈の結論とは分ける。

**推奨は、ここで追加結果をチェックポイントへ整理すること。** 校了本文への追記は次の改訂で判断できる。速度改善を続ける場合の次候補は、新blockの圧縮・連結に絞った部品診断。現時点では新実験や実装変更を実行していない。

[固定設計](../plans/loop-kv-nb06-design-v0.1-2026-10-08.md) / [原結果](../runs/{d['run_id']}/summary.json) / [再集計JSON]({BASE}.json) / [再集計コード](../scripts/analyze_loop_kv_nb06_results.py) / [照合記録](../throughline/audits/loop-kv-nb06-results-v0.1-2026-10-08/verification-summary.json)
'''
    (out/(BASE+'.md')).write_text(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data=analyze(a.run)
    (a.output/(BASE+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    plot(data,a.output/'figures'/(BASE+'.png'));report(data,a.output)
    print(json.dumps({'nll_components_checked':data['nll_components_checked'],'max_nll_component_difference':data['max_nll_component_difference'],
        'relative_ppl_percent':data['quality']['full_256']['relative_ppl_percent_vs_native'],
        'timing_256':data['timing']['cuda_ms']['256'],'storage_reduction_percent':data['storage_reduction_percent']},indent=2))

if __name__=='__main__':main()
