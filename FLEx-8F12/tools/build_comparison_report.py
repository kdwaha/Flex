"""Build a self-contained HTML snapshot from existing diagnostics (no training)."""
import argparse
import base64
import csv
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'untrained':'Untrained baseline', 'scaffold_reset':'SCAFFOLD-reset', 'frlora':'FRLoRA',
         'frlora_scaffold':'FRLoRA + SCAFFOLD', 'fedavg':'FedAvg + AdamW',
         'fedsa':'FedSA-LoRA (global proxy)', 'scaffold':'SCAFFOLD', 'pflalign':'pFLAlign full (erf)'}
NAMES.update({f'pflalign_{key}': label for key, label in {
    'no_preconditioner': 'pFLAlign: P=1', 'no_correction': 'pFLAlign: gamma=0',
    'constant_gamma': 'pFLAlign: gamma=0.5', 'hard_gamma': 'pFLAlign: hard gamma',
    'constant_gamma_one': 'pFLAlign: gamma=1',
    'no_personalization': 'pFLAlign: Delta=0', 'sgd': 'FedAvg + SGD (P=1, Delta=0)',
}.items()})
METRICS = [('test_loss','Test loss ↓'), ('perplexity','PPL ↓'),
           ('token_accuracy','Token accuracy ↑'), ('exact_match','Exact match ↑'),
           ('rougeL_f1','ROUGE-L ↑'), ('bleu4_pct','BLEU-4 ↑')]
HESSIAN = [('hessian_lambda_max','λ max'), ('hessian_lambda_min','λ min'),
           ('hessian_trace','Trace'), ('hessian_top1_top2','λ1 / λ2'),
           ('hessian_max_abs_min','λ max / |λ min|')]


def line_style(algorithm, sparse=False):
    """Stable method identity across panels, including monochrome prints."""
    if algorithm == 'untrained':
        return dict(color='black', linestyle='--', linewidth=1.8)
    index = list(NAMES).index(algorithm) - 1
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    return dict(color=plt.get_cmap('tab10')(index % 10),
                linestyle=['-', '--', '-.', ':'][(index // len(markers)) % 4],
                marker=markers[index % len(markers)], markersize=5,
                markerfacecolor='none', markeredgewidth=1.1, linewidth=1.6,
                markevery=(index % 30, 55) if sparse else 1)


def comparison_legend(fig):
    # Keep the 15-method key outside the data rather than over the first panel.
    entries = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles))
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if entries:
        fig.legend(entries.values(), entries.keys(), loc='lower center',
                   bbox_to_anchor=(0.5, 0), ncol=3, fontsize=8, handlelength=3.5)
    fig.tight_layout(rect=(0, 0.15 if entries else 0, 1, 0.96 if fig._suptitle else 1))


def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+\
           '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def number(x):
    return '—' if x is None else f'{x:.4f}'


def build(root):
    runs = {}
    for path in sorted(root.rglob('diagnostics.json')):
        config = json.loads((path.parent/'args.json').read_text())
        records = json.loads(path.read_text())
        for record in records:
            maximum, minimum = record.get('hessian_lambda_max'), record.get('hessian_lambda_min')
            if maximum is not None and minimum is not None:
                record['hessian_max_abs_min'] = maximum / abs(minimum) if minimum != 0 else None
        alg = 'untrained' if config['script_args'].get('baseline_eval_only') else config['fed_args']['fed_alg']
        variant = config['fed_args'].get('pflalign_variant', 'full')
        if alg == 'pflalign' and variant != 'full':
            alg += '_' + variant
        runs[alg] = {'records':records, 'path':path, 'done':(path.parent/'metrics.json').exists()}
    out=root/'report'
    out.mkdir(exist_ok=True)
    parts=['<h1>연합학습 통합 비교</h1>', '<p>생성 시각: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' KST · 현재 저장 결과의 스냅샷</p>',
           '<p>TinyLlama 1.1B · Dolly 4 task clients · 30 rounds × 5 local epochs · LR 2e-4 constant · GPU 0 · seed 2025 · 생성 최대 500토큰.</p>',
           '<p>로컬: 학습 직후 자기 태스크 test. 글로벌: 집계 후 전체 test. Test는 학습과 레코드 단위로 분리된 14개(4/3/4/3)이며 학습은 62개입니다. 단일 seed·소규모 결과입니다.</p>']
    statuses=[]
    for alg,name in NAMES.items():
        run=runs.get(alg)
        r=run['records'] if run else []
        last=max((x['round'] for x in r),default=0)
        ev=max((x['round'] for x in r if x['kind']=='evaluation' and x['client']==-1),default=0)
        statuses.append([name,'완료' if run and run['done'] else '진행 중' if r else '대기',last,0 if alg=='untrained' and r else ev or '—'])
    parts+=['<h2>실행 상태</h2>',table(['방법','상태','최근 기록 라운드','글로벌 평가 라운드'],statuses),
            '<p>아래 표는 완료된 30라운드 결과와 학습 전 baseline(0라운드)을 비교합니다. 진행 중인 방법은 곡선에 실제 측정 시점까지만 표시합니다. Baseline은 추가 학습 없이 같은 test·프롬프트·생성 500 조건으로 평가했습니다.</p>']
    complete={a:v for a,v in runs.items() if v['done'] and any(x['kind']=='evaluation' and x['client']==-1 and x['round']==(0 if a=='untrained' else 30) for x in v['records'])}
    complete={a:complete[a] for a in NAMES if a in complete}
    csvrows=[]
    for phase in ['global','local']:
        rows=[]
        for alg,run in complete.items():
            ev=[x for x in run['records'] if x['kind']=='evaluation' and x['round']==(0 if alg=='untrained' else 30) and (x['client']==-1 if phase=='global' else x['client']>=0)]
            vals={k:float(np.mean([x[k] for x in ev])) for k,_ in METRICS}
            rows.append([NAMES[alg]]+[number(vals[k]) for k,_ in METRICS])
            csvrows.append(dict(algorithm=alg,phase=phase,round=0 if alg=='untrained' else 30,**vals))
        parts+=['<h2>30라운드 '+('글로벌 · 전체 test' if phase=='global' else '로컬 · 클라이언트 Macro 평균')+'</h2>',
                table(['방법']+[label for _,label in METRICS],rows)]
    parts+=['<p>로컬 표는 각 클라이언트 지표의 산술평균입니다(PPL도 산술평균). 글로벌 PPL은 전체 정답 토큰의 NLL을 합산한 뒤 exp를 적용합니다. Accuracy·EM·ROUGE-L은 0–1, BLEU는 0–100입니다. FedSA 글로벌 곡선은 공유 A + 샘플가중 평균 B의 비교용 proxy이며 공식 개인화 성능과 구분합니다.</p>']
    rows=[]
    for alg,run in complete.items():
        for r in run['records']:
            if r['kind']=='evaluation' and r['round']==(0 if alg=='untrained' else 30) and r['client']>=0:
                rows.append([NAMES[alg],f'C{r["client"]}',r['task']]+[number(r[k]) for k,_ in METRICS])
    parts+=['<details><summary>30라운드 클라이언트·태스크별 수치 펼치기</summary>',table(['방법','클라이언트','태스크']+[v for _,v in METRICS],rows),'</details>']

    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    def figure(fig,name):
        comparison_legend(fig)
        fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=120,bbox_inches='tight')
        parts.append('<img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">')
        plt.close(fig)
    for phase in ['global','local']:
        parts+=['<h2>'+('글로벌' if phase=='global' else '로컬 Macro 평균')+' 성능 추이</h2>']
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for alg,run in runs.items():
            ev=[r for r in run['records'] if r['kind']=='evaluation' and (r['client']==-1 if phase=='global' else r['client']>=0)]
            rounds=sorted({r['round'] for r in ev})
            if phase=='local':
                rounds=[t for t in rounds if len([r for r in ev if r['round']==t])==4]
            for ax,(key,label) in zip(axes.flat,METRICS):
                if alg=='untrained':
                    ax.axhline(np.mean([r[key] for r in ev]), **line_style(alg), label=NAMES[alg])
                else:
                    ax.plot(rounds,[np.mean([r[key] for r in ev if r['round']==t]) for t in rounds], **line_style(alg), label=NAMES[alg])
                ax.set(title=label,xlabel='Global round',xticks=[5,10,15,20,25,30])
                ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        figure(fig,phase+'_performance')
    parts+=['<h2>학습 loss · 클라이언트별 방법 비교</h2>']
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for c,ax in enumerate(axes.flat):
        for alg,run in runs.items():
            rows=[r for r in run['records'] if r['kind']=='local_update' and r['client']==c and r.get('train_loss') is not None]
            ax.plot([r['round'] for r in rows],[r['train_loss'] for r in rows], **line_style(alg), label=NAMES[alg])
        ax.set(title=f'Client {c}',xlabel='Global round',ylabel='Train loss'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    figure(fig,'client_training_loss')
    parts+=['<h2>30라운드 Hessian 수치</h2>', '<p>LoRA A/B completion-loss Hessian의 추정값입니다. Lanczos 20 steps × 2 probes, Hutchinson trace 4 probes. Trace는 표본 오차가 있으며 음수가 될 수 있습니다. 전체 모델 Hessian이 아닙니다.</p>']
    rows=[]
    for alg,run in complete.items():
        for r in run['records']:
            if r['kind']=='evaluation' and 'hessian_trace' in r:
                rows.append([NAMES[alg], 'Global' if r['client']==-1 else f'C{r["client"]}']+[number(r.get(k)) for k,_ in HESSIAN]+[number(r.get('hessian_trace_se'))])
    parts.append(table(['방법','모델']+[v for _,v in HESSIAN]+['Trace SE'],rows))
    parts+=['<h2>Hessian spectrum density · 방법 간 비교</h2>']
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for c,ax in zip([0,1,2,3,-1],axes.flat):
        selected=[(a,r) for a,run in complete.items() for r in run['records'] if r['kind']=='evaluation' and r['client']==c and 'hessian_spectrum' in r]
        nodes=[n for _,r in selected for s in r['hessian_spectrum'] for n in s['nodes']]
        if not nodes: continue
        lo,hi=min(nodes),max(nodes); bw=max((hi-lo)*.02,1e-6); grid=np.linspace(lo-3*bw,hi+3*bw,700)
        for a,r in selected:
            density=sum((np.exp(-.5*((grid[:,None]-np.array(s['nodes']))/bw)**2)*np.array(s['weights'])).sum(1) for s in r['hessian_spectrum'])/(len(r['hessian_spectrum'])*bw*np.sqrt(2*np.pi))
            ax.plot(grid,density, **line_style(a, sparse=True), label=NAMES[a])
        ax.set(title='Global / all test' if c==-1 else f'Client {c}',xlabel='Eigenvalue',ylabel='SLQ density'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7); axes.flat[-1].axis('off')
    figure(fig,'spectrum_comparison')
    parts+=['<h2>FedTorch 연합학습 지표 · 방법 간 비교</h2>']
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,key in zip(axes.flat,['consistency','drift_diversity','global_update_norm','correction_norm']):
        for alg,run in runs.items():
            rows=[r for r in run['records'] if r['kind']=='server_update' and r.get(key) is not None]
            ax.plot([r['round'] for r in rows],[r[key] for r in rows], **line_style(alg), label=NAMES[alg])
        ax.set(title=key,xlabel='Global round'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7); figure(fig,'federated_comparison')
    parts+=['<p>위 norm/drift는 LoRA factor 공간 기준입니다. FRLoRA의 base residual 변화는 포함하지 않으며, factor reset 때문에 글로벌 factor 변화가 0이면 drift diversity는 정의되지 않아 생략됩니다.</p>']
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Federated comparison</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#182334;background:#fafbfd}h1,h2{color:#123b5d}h2{margin-top:36px}p{line-height:1.7}img{width:100%;background:white;border-radius:10px;margin:12px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:10px;border-bottom:1px solid #dbe1e8;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#e8f0f7}summary{cursor:pointer;padding:14px;background:#e8f0f7}</style>'+''.join(parts)+'</html>'
    (out/'comparison.html').write_text(document)
    with (out/'round30_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['algorithm','phase','round']+[k for k,_ in METRICS]);writer.writeheader();writer.writerows(csvrows)
    # A plain Markdown entry point plus per-client performance figures.
    md=['# 연합학습 통합 비교', '',
        '각 클라이언트의 로컬 학습 직후 모델을 자기 태스크 test에서 평가했습니다. '
        '학습 전 baseline은 0라운드, 학습 방법은 30라운드 결과입니다.', '',
        'Loss·PPL은 낮을수록 좋고, 나머지는 높을수록 좋습니다. '
        'Token accuracy·Exact match·ROUGE-L은 0–1, BLEU는 0–100입니다.', '',
        'TinyLlama 1.1B · 30 rounds × 5 local epochs · constant LR 2e-4 · 생성 최대 500토큰. '
        'Test는 학습과 분리된 총 14개이며 단일 seed 결과입니다.', '',
        '[Hessian 비교표·그림](hessian_comparison.md) · [전체 로컬 평균 곡선](local_performance.png)', '']
    md += ['## 실행 상태 (보고서 갱신 시점)', '',
           '아래 성능표에는 완료된 조건만 포함됩니다. 각 조건 종료 후 자동 갱신됩니다.', '',
           '| 방법 | 상태 | 마지막 기록 round | 글로벌 평가 round |', '|---|---|---:|---:|']
    md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in statuses]
    md += ['']
    for phase,title in [('global','글로벌 전체 test'),('local','로컬 Macro 평균')]:
        md += [f'## {title}', '', '| 방법 | '+' | '.join(v for _,v in METRICS)+' |',
               '|---|'+'---:|'*len(METRICS)]
        for row in csvrows:
            if row['phase']==phase:
                md.append('| '+NAMES[row['algorithm']]+' | '+' | '.join(number(row[k]) for k,_ in METRICS)+' |')
        md += ['', '![성능 비교]('+('global' if phase=='global' else 'local')+'_performance.png)', '']
    md += ['로컬 평균은 클라이언트별 지표의 산술평균입니다. 글로벌 PPL은 전체 정답 토큰 NLL로 계산합니다. '
           'FedSA의 글로벌 모델은 공유 A와 가중 평균 B로 만든 비교용 proxy입니다.', '']
    client_csv=[]
    for c in range(4):
        finals=[]
        for alg,run in complete.items():
            r=next(x for x in run['records'] if x['kind']=='evaluation' and x['client']==c
                   and x['round']==(0 if alg=='untrained' else 30))
            finals.append((alg,r))
            client_csv.append(dict(algorithm=alg,client=c,task=r['task'],round=r['round'],
                                   **{k:r[k] for k,_ in METRICS}))
        task=finals[0][1]['task']; examples=finals[0][1]['examples']
        md += [f'## Client {c} — {task}', '', f'Test {examples}개.', '',
               '| 방법 | '+' | '.join(label for _,label in METRICS)+' |',
               '|---|'+'---:|'*len(METRICS)]
        for alg,r in finals:
            label=NAMES[alg].replace(' (global proxy)','')
            md.append('| '+label+' | '+' | '.join(number(r[k]) for k,_ in METRICS)+' |')
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for alg,run in runs.items():
            ev=[r for r in run['records'] if r['kind']=='evaluation' and r['client']==c]
            for ax,(key,label) in zip(axes.flat,METRICS):
                if alg=='untrained':
                    ax.axhline(ev[0][key], **line_style(alg), label='Untrained baseline')
                else:
                    ax.plot([r['round'] for r in ev],[r[key] for r in ev], **line_style(alg),
                            label=NAMES[alg].replace(' (global proxy)',''))
                ax.set(title=label,xlabel='Global round',xticks=[5,10,15,20,25,30]);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'Client {c}: {task} — own-task heldout performance')
        comparison_legend(fig)
        for ext in ['png','pdf']:
            fig.savefig(out/f'client_{c}_performance.{ext}',dpi=140,bbox_inches='tight')
        plt.close(fig)
        md += ['', f'![Client {c} 성능 곡선](client_{c}_performance.png)', '']
    md += ['## 전체 비교 그림', '', '[글로벌 성능](global_performance.png) · '
           '[로컬 평균](local_performance.png) · [클라이언트별 학습 loss](client_training_loss.png) · '
           '[연합학습 지표](federated_comparison.png) · [Hessian spectrum](spectrum_comparison.png)', '']
    generation_keys = ['eos_rate', 'length_limit_rate', 'mean_generated_tokens',
                       'repeated_4gram_fraction', 'bleu_brevity_penalty']
    generation_rows = []
    for a, run in complete.items():
        for r in run['records']:
            if r['kind'] == 'evaluation' and r['round'] == (0 if a == 'untrained' else 30) and 'eos_rate' in r:
                generation_rows.append(dict(algorithm=a, client=r['client'], task=r['task'],
                    **{k: r.get(k) for k in generation_keys},
                    **{f'bleu_precision_{i}_raw': r.get(f'bleu_precision_{i}_raw') for i in range(1, 5)}))
    if generation_rows:
        md += ['## EOS·반복·BLEU 길이 패널티', '',
               'EOS 이후 배치 padding은 생성 길이에서 제외합니다. 반복률은 예제별 중복 4-gram 비율의 평균입니다. '
               'BP는 corpus BLEU의 brevity penalty입니다. 상세 n-gram precision은 generation_comparison.csv에 저장합니다.', '',
               '| 방법 | 평가 | EOS 비율 | 길이제한 비율 | 평균 생성 토큰 | 반복률 | BLEU BP |',
               '|---|---|---:|---:|---:|---:|---:|']
        for r in generation_rows:
            md.append('| ' + NAMES[r['algorithm']] + ' | ' + ('Global' if r['client'] == -1 else f'C{r["client"]}')
                      + ' | ' + ' | '.join(number(r[k]) for k in generation_keys) + ' |')
        md += ['', '이번 loss에는 실제 response EOS가 포함되며 prompt와 padding은 제외됩니다. '
               'EOS를 제외했던 이전 loss/PPL과 동일한 지표 정의가 아니므로 새 baseline과 비교해야 합니다.', '',
               'SCAFFOLD correction_norm은 control-variate 통계이며 pFLAlign의 gamma·Delta correction 크기를 뜻하지 않습니다.', '']
        with (out/'generation_comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(generation_rows[0]))
            writer.writeheader(); writer.writerows(generation_rows)
    hmd=['# 30라운드 Hessian 비교', '', '[전체 성능 비교로 돌아가기](comparison.md)', '',
         'LoRA A/B에 대한 completion-token 평균 NLL의 Hessian 추정값입니다. '
         '로컬은 학습 직후 자기 태스크 test, 글로벌은 집계 후 전체 test에서 측정했습니다.', '',
         'Lanczos 20 steps × 2 probes, Hutchinson trace 4 probes. '
         '최대·최소·두 번째 고유값은 Ritz 추정값이며, λ1/λ2는 대수적으로 가장 큰 두 고유값의 비율입니다. '
         'λ max / |λ min|는 최대 고유값을 최소 고유값의 절댓값으로 나눈 값입니다. '
         '최소 고유값이 0이면 정의되지 않아 —로 표시하며, 0에 가까우면 비율이 민감해집니다. '
         'Trace SE는 trace 추정의 표준오차입니다. 숫자가 작다고 자동으로 성능이 좋다는 뜻은 아닙니다.', '',
         '학습 전 baseline은 Hessian을 측정하지 않아 아래 비교에서 제외했습니다. '
         'FedSA의 글로벌 행은 공유 A + 가중 평균 B proxy입니다.', '']
    hcsv=[]
    columns=HESSIAN+[('hessian_lambda_2','λ2'),('hessian_trace_se','Trace SE')]
    for c in [-1,0,1,2,3]:
        selected=[(a,r) for a,run in complete.items() for r in run['records']
                  if r['kind']=='evaluation' and r['round']==30 and r['client']==c and 'hessian_trace' in r]
        title='글로벌 · 전체 test' if c==-1 else f'Client {c} · '+(selected[0][1]['task'] if selected else '')
        hmd += ['## '+title, '', '| 방법 | '+' | '.join(v for _,v in columns)+' |',
                '|---|'+'---:|'*len(columns)]
        for a,r in selected:
            label=NAMES[a] if c==-1 else NAMES[a].replace(' (global proxy)','')
            hmd.append('| '+label+' | '+' | '.join('—' if r.get(k) is None else f'{r[k]:.6g}' for k,_ in columns)+' |')
            hcsv.append(dict(algorithm=a,client=c,round=30,**{k:r.get(k) for k,_ in columns}))
        fig,axes=plt.subplots(3,2,figsize=(13,12))
        for ax,(key,label) in zip(axes.flat,HESSIAN):
            valid=[(a,r) for a,r in selected if r.get(key) is not None]
            bars = ax.barh([NAMES[a].replace(' (global proxy)',' *') if c==-1 else NAMES[a].replace(' (global proxy)','') for a,r in valid],
                    [r[key] for a,r in valid], color=[line_style(a)['color'] for a,r in valid],
                    edgecolor='#333333', linewidth=0.5)
            for bar, (a, _) in zip(bars, valid):
                bar.set_hatch(['/', '\\', '|', '-', '+', 'x', '.', 'o', '*', '//', '\\\\', '||', 'xx', '..'][list(NAMES).index(a)-1])
            ax.invert_yaxis();ax.axvline(0,color='gray',linewidth=.7)
            ax.set(title=label,xlabel='Estimate');ax.grid(axis='x',alpha=.2)
        axes.flat[-1].axis('off')
        fig.suptitle('Round 30 Hessian: '+('Global (FedSA: proxy)' if c==-1 else f'Client {c}'))
        fig.tight_layout()
        name='hessian_global' if c==-1 else f'hessian_client_{c}'
        for ext in ['png','pdf']:
            fig.savefig(out/f'{name}.{ext}',dpi=140,bbox_inches='tight')
        plt.close(fig)
        hmd += ['',f'![{title}]({name}.png)','']
    hmd += ['## Spectrum density', '', '방법 간 동일한 smoothing bandwidth를 사용한 SLQ 밀도 추정입니다.', '',
            '![Hessian spectrum 비교](spectrum_comparison.png)', '']
    (out/'hessian_comparison.md').write_text('\n'.join(hmd))
    with (out/'hessian_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['algorithm','client','round']+[k for k,_ in columns])
        writer.writeheader();writer.writerows(hcsv)
    md += ['## Hessian 비교', '', '[글로벌·클라이언트별 수치표와 그림](hessian_comparison.md)', '',
           '![글로벌 Hessian 비교](hessian_global.png)', '']
    (out/'comparison.md').write_text('\n'.join(md))
    with (out/'client_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['algorithm','client','task','round']+[k for k,_ in METRICS])
        writer.writeheader();writer.writerows(client_csv)
    print(out/'comparison.md')
    print(out/'hessian_comparison.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);build(p.parse_args().root)
