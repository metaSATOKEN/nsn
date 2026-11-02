"""Build both papers’ continuation tables and numerical audits offline."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import shutil
import math

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|',
                     '|'+'|'.join(['---']+['---:']*(len(header)-1))+'|']
                    + ['|'+'|'.join(map(str,row))+'|' for row in rows])

def pct(x):
    return f'{x:+.3f}%'.replace('-', '−')

def build(root, output, spec_path, integrated_base, loop_base):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Use an output directory outside the evidence tree.')
    spec=json.loads(spec_path.read_text())
    base_data = {
        'papers/nsn-integrated/paper-data-v0.4-2026-10-05.json': integrated_base,
        'papers/loop-kv/paper-data-v0.2-2026-10-03.json': loop_base,
    }
    checked={}
    for rel, expected in spec['inputs'].items():
        actual=digest(base_data.get(rel, root/rel))
        if actual != expected:
            raise ValueError(f'Input hash mismatch: {rel}')
        checked[rel]=actual
    s=json.loads((root/f"runs/{spec['run_id']}/summary.json").read_text())
    report=json.loads((root/'reports/nb06-results-v0.1-2026-10-08.json').read_text())
    assert s['status']=='PASS_MODEL_CONNECTION'
    assert (s['material_count'],s['trajectories'],s['completed_forward_calls'],s['actual_q_captures'])==(4,60,15420,32)
    assert len(report['books'])==4
    assert report['summary_sha256']==checked[f"runs/{spec['run_id']}/summary.json"]
    diag=[x for b in s['books'] for x in b['actual_q']]
    assert len(diag)==32 and all(x['pass_all'] for x in diag)
    assert all(all(x['tile_fused_restore_equal'].values()) for x in diag)
    assert all(x['model_replay_equal'] and x['output_rounding_equal'] and all(x['trace_normal_equal'].values()) for x in diag)
    raw_rows=s['raw_logit_trajectories']*8
    assert raw_rows==384
    # All quality windows: never replace the per-text checks with an aggregate.
    for b in report['books']:
        for q in b['quality'].values():
            for arm in ['PACKED_SDPA','GQA_FUSED']:
                assert q['relative_ppl_percent_vs_native'][arm]<=5
                delta=q['mean_nll'][arm]-q['mean_nll']['NATIVE']
                assert math.isclose(100*math.expm1(delta),q['relative_ppl_percent_vs_native'][arm],abs_tol=1e-11)
    totals={}
    for arm in ['NATIVE','PACKED_SDPA','GQA_FUSED']:
        vals={b['terminal_kv_bytes'][arm] for b in s['books']}
        assert len(vals)==1
        totals[arm]=vals.pop()
    reduction=100*(1-totals['GQA_FUSED']/totals['NATIVE'])
    assert math.isclose(reduction,report['storage_reduction_percent'],abs_tol=1e-12)
    mem={}
    for arm in totals:
        rows=[row for b in s['books'] for row in b['memory_peak_bytes'][arm]['256']]
        assert len(rows)==12
        assert all(row['kv_storage_bytes']==totals[arm] for row in rows)
        values={}
        for phase in ['decode_peak','full_phase_peak']:
            values[phase]={}
            for kind in ['allocated','reserved']:
                distinct={r[phase][kind] for r in rows}
                assert len(distinct)==1
                values[phase][kind]=distinct.pop()
        mem[arm]=values
    maxima={
      'k_reconstruction':max(x['restoration']['k']['max_abs'] for x in diag),
      'v_reconstruction':max(x['restoration']['v']['max_abs'] for x in diag),
      'score':max(x['score']['max_abs'] for x in diag),
      'context':max(x['context']['max_abs'] for x in diag),
      'sdpa':max(y['max_abs'] for x in diag for y in x['sdpa'].values()),
    }
    numeric={
      'run_id':spec['run_id'], 'summary_sha256':report['summary_sha256'],
      'actual_q_count':32, 'max_abs':maxima, 'all_fixed_budgets_pass':True,
      'references':'The same quantized-input references and budgets as NB05; manuscript Table B1.',
      'bridge_fused_kv_bit_equal':True,
      'forward_calls':15420, 'raw_logit_rows':384,
      'evidence_scope':'All-position NLL components and logit hashes; raw FP32 logits only at 8 fixed steps in 48 non-warmup trajectories.',
      'max_nll_component_difference':report['max_nll_component_difference'],
    }
    qrows=[]
    for b in report['books']:
        q=b['quality']
        qrows.append([b['book_id'],pct(q['full_256']['relative_ppl_percent_vs_native']['GQA_FUSED']),
                      pct(q['block_4']['relative_ppl_percent_vs_native']['GQA_FUSED'])])
    agg=[pct(report['quality'][k]['relative_ppl_percent_vs_native']['GQA_FUSED']) for k in ['full_256','block_4']]
    fragments={
      'QUALITY_TABLE_JA':table(['PG19 ID','全256予測','終盤64予測'],qrows+[['4文章集約']+agg]),
      'QUALITY_TABLE_EN':table(['PG19 ID','All 256 predictions','Final 64 predictions'],qrows+[['Four-text aggregate']+agg]),
    }
    trows=[]
    for step in ['8','64','128','256']:
        x=report['timing']['cuda_ms'][step]
        for metric in ['decode_ms','phase_sum_ms','forward_ms']:
            for other in ['NATIVE','PACKED_SDPA']:
                recomputed=statistics.median(b['costs']['cuda_ms'][step]['ratios'][metric][other]['median_ratio'] for b in report['books'])
                assert math.isclose(recomputed,x[metric][other],abs_tol=1e-14)
        trows.append([step,pct(100*(x['decode_ms']['NATIVE']-1)),
                     pct(100*(x['phase_sum_ms']['NATIVE']-1)),
                     pct(100*(x['decode_ms']['PACKED_SDPA']-1))])
    fragments['TIMING_TABLE_JA']=table(['継続step','更新込みdecode／非圧縮','初期費用込み／非圧縮','更新込みdecode／全量復元'],trows)
    fragments['TIMING_TABLE_EN']=table(['Continuation steps','Decode + updates / uncompressed','Including initialization / uncompressed','Decode + updates / full reconstruction'],trows)
    nr_ja=[['K復元／凍結復元器',maxima['k_reconstruction']],
           ['V復元／凍結復元器',maxima['v_reconstruction']],
           ['QK score／当該復元KのFP64参照',format(maxima['score'],'.8g')],
           ['context／保存score・当該VのFP64参照',format(maxima['context'],'.8g')],
           ['両SDPA経路／各経路のFP64参照',format(maxima['sdpa'],'.8g')]]
    nr_en=[['K reconstruction / frozen reconstruction',maxima['k_reconstruction']],
           ['V reconstruction / frozen reconstruction',maxima['v_reconstruction']],
           ['QK score / FP64 using that reconstructed K',format(maxima['score'],'.8g')],
           ['Context / FP64 using saved scores and that V',format(maxima['context'],'.8g')],
           ['Both SDPA routes / route-specific FP64 reference',format(maxima['sdpa'],'.8g')]]
    fragments['NUMERIC_TABLE_JA']=table(['検査対象／参照','最大絶対差'],nr_ja)
    fragments['NUMERIC_TABLE_EN']=table(['Check / reference','Maximum absolute difference'],nr_en)
    compact_books=[]
    for b in report['books']:
        ratio=b['costs']['cuda_ms']['256']['ratios']['phase_sum_ms']['NATIVE']
        compact_books.append({'book_id':b['book_id'],'quality':b['quality'],'total_time_256_vs_native':ratio})
    payload={'run_id':spec['run_id'], 'source_sha256':checked,
             'quality':report['quality'],'timing':report['timing'],
             'mean_additive_cost_cuda_ms':report['mean_additive_cost_cuda_ms'],
             'terminal_kv_bytes':totals,'storage_reduction_percent':reduction,
             'memory_peak_bytes':mem,'books':compact_books,'numerical_audit':numeric,
             'protocol':{'prefill_tokens':128,'continuation_predictions':256,'blocks':64,
                         'post_decode_update_steps':[64,128,192,256],
                         'updates_consumed_by_subsequent_attention':3,
                         'terminal_compressed_tokens':320,'terminal_bf16_tail_tokens':64,
                         'book_count':4,'main_replicates':3,'new_books_relative_to_nb05':0,
                         'primary_time':'median within text of three paired ratios, then median across four texts',
                         'cost_decomposition':'post hoc arithmetic means across twelve primary measurements'},
             'generated_tables':fragments}
    # Published JSON includes the original document identifiers as provenance.
    # File verification above covers the numerical inputs used by this builder.
    payload['source_sha256'] = {**checked, **spec['provenance_only']}
    for folder,oldname,newname in [
        ('nsn-integrated','paper-data-v0.4-2026-10-05.json','paper-data-v0.5-2026-10-08.json'),
        ('loop-kv','paper-data-v0.2-2026-10-03.json','paper-data-v0.3-2026-10-08.json')]:
        baseline=json.loads(base_data[f'papers/{folder}/{oldname}'].read_text())
        # Preserve previous measurements; edition metadata and new data are separate.
        assert 'nb06_continuation' not in baseline
        inherited_metadata={k:baseline[k] for k in ['version','date','scope'] if k in baseline}
        baseline['nb06_continuation']=payload
        baseline['supplement_edition']={'date':spec['date'],'base_data':oldname,
                                       'base_sha256':checked[f'papers/{folder}/{oldname}'],
                                       'inherited_metadata':inherited_metadata}
        baseline['version']='0.5' if folder=='nsn-integrated' else '0.3'
        baseline['date']=spec['date']
        baseline['scope']='Retained earlier measurements plus NB06 continuation evidence; no new model execution during manuscript preparation.'
        write_json(output/'papers'/folder/newname,baseline)
        write_json(output/'papers'/folder/'numerical-audit-nb06-v0.1-2026-10-08.json',numeric)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--integrated-base', type=Path, required=True)
    parser.add_argument('--loop-base', type=Path, required=True)
    args = parser.parse_args()
    build(args.research_root, args.output_dir, Path(__file__).with_name('nb06_table_sources.json'),
          args.integrated_base, args.loop_base)
