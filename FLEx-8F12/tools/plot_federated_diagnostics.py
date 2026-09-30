"""Plot stored periodic diagnostics without importing training code."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_run(path):
    records = json.loads(path.read_text())
    out = path.parent/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'font.size': 10})
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)

    metrics = [('test_loss', 'Test loss'), ('perplexity', 'Test PPL'),
               ('token_accuracy', 'Token accuracy'), ('exact_match', 'Exact match'),
               ('rougeL_f1', 'ROUGE-L'), ('bleu4_pct', 'BLEU-4')]
    hmetrics = [('hessian_lambda_max','Largest eigenvalue'), ('hessian_lambda_min','Smallest eigenvalue'),
                ('hessian_trace','Hessian trace'), ('hessian_top1_top2','Top 1 / top 2')]
    for name, specs in [('test_metrics',metrics), ('hessian_metrics',hmetrics)]:
        fig, axes = plt.subplots((len(specs)+1)//2, 2, figsize=(12, 3.5*((len(specs)+1)//2)), squeeze=False)
        for ax, (key, title) in zip(axes.flat, specs):
            for client in [0,1,2,3,-1]:
                rows = [r for r in records if r['kind']=='evaluation' and r['client']==client and r.get(key) is not None]
                if rows:
                    label = 'Global (all test)' if client==-1 else f'C{client}: {rows[0]["task"]}'
                    ax.plot([r['round'] for r in rows], [r[key] for r in rows], marker='o',
                            label=label, linewidth=2.5 if client==-1 else 1.4)
            # Macro is explicitly client metric average, not pooled global score.
            local = [r for r in records if r['kind']=='evaluation' and r['client']>=0 and r.get(key) is not None]
            rounds = sorted({r['round'] for r in local})
            ax.plot(rounds,[np.mean([r[key] for r in local if r['round']==t]) for t in rounds],
                    '--', color='black', label='Local macro mean')
            ax.set(title=title, xlabel='Global round')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        save(fig,name)

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for c in range(4):
        for ax,kind in zip(axes,['local_update','epoch']):
            rows=[r for r in records if r['kind']==kind and r['client']==c and r.get('train_loss') is not None]
            x=[r['round'] if kind=='local_update' else (r['round']-1)*5+r['local_epoch'] for r in rows]
            ax.plot(x,[r['train_loss'] for r in rows],label=f'Client {c}')
            ax.set(xlabel='Global round' if kind=='local_update' else 'Cumulative local epoch', ylabel='Train loss')
            ax.grid(alpha=.2)
    axes[0].legend()
    save(fig,'training_curves')

    specs=['cosine_similarity','weight_norm','local_update_norm','global_distance','correction_norm','consistency','drift_diversity','global_update_norm']
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for ax,key in zip(axes.flat,specs):
        for c in [0,1,2,3,-1]:
            rows=[r for r in records if r['kind'] in ['local_update','server_update'] and r['client']==c and r.get(key) is not None]
            if rows:
                ax.plot([r['round'] for r in rows],[r[key] for r in rows],label='Global' if c==-1 else f'C{c}')
        ax.set(title=key, xlabel='Global round')
        ax.grid(alpha=.2)
    axes.flat[1].legend()
    save(fig,'federated_metrics')

    fig, axes = plt.subplots(3,2,figsize=(12,10))
    for ax,c in zip(axes.flat,[0,1,2,3,-1]):
        rows=[r for r in records if r['kind']=='evaluation' and r['client']==c and 'hessian_spectrum' in r]
        all_nodes=[node for row in rows for s in row['hessian_spectrum'] for node in s['nodes']]
        if not all_nodes:
            continue
        lo,hi=min(all_nodes),max(all_nodes)
        bandwidth=max((hi-lo)*.03,1e-6)
        grid=np.linspace(lo-3*bandwidth,hi+3*bandwidth,500)
        for row in rows:
            specs=row['hessian_spectrum']
            density=np.zeros_like(grid)
            for s in specs:
                nodes=np.array(s['nodes']); weights=np.array(s['weights'])
                density += (np.exp(-.5*((grid[:,None]-nodes)/bandwidth)**2)*weights).sum(1)/(bandwidth*np.sqrt(2*np.pi)*len(specs))
            ax.plot(grid,density,label=f'R{row["round"]}')
        ax.set(title=f'Client {c}' if c>=0 else 'Global: all test',xlabel='Eigenvalue',ylabel='SLQ density (Gaussian smoothing)')
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    save(fig,'hessian_spectrum_density')
    print(out)


def plot_comparison(paths, root):
    if len(paths)<2:
        return
    metrics=['test_loss','perplexity','token_accuracy','exact_match','rougeL_f1','bleu4_pct',
             'hessian_lambda_max','hessian_lambda_min','hessian_trace','hessian_top1_top2']
    out=root/'comparison_figures'
    out.mkdir(exist_ok=True)
    for phase in ['global','local_macro']:
        fig,axes=plt.subplots(5,2,figsize=(12,16))
        for path in paths:
            algorithm=json.loads((path.parent/'args.json').read_text())['fed_args']['fed_alg']
            records=[r for r in json.loads(path.read_text()) if r['kind']=='evaluation'
                     and (r['client']==-1 if phase=='global' else r['client']>=0)]
            for ax,key in zip(axes.flat,metrics):
                valid=[r for r in records if r.get(key) is not None]
                rounds=sorted({r['round'] for r in valid})
                ax.plot(rounds,[np.mean([r[key] for r in valid if r['round']==t]) for t in rounds],
                        marker='o',label=algorithm)
                ax.set(title=key,xlabel='Global round')
                ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('Global model / whole test' if phase=='global' else 'Local models / task test: macro mean')
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(out/f'{phase}.{ext}',bbox_inches='tight')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    paths=[args.root] if args.root.is_file() else sorted(args.root.rglob('diagnostics.json'))
    for path in paths:
        plot_run(path)
    if args.root.is_dir():
        plot_comparison(paths,args.root)
