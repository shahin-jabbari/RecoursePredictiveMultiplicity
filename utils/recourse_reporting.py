"""Dataset-specific publication exports using V4's existing seed-level intervals.

Reports consume CI endpoints; they never estimate uncertainty from summary rows.
CSV files preserve numeric data and missing values. Compact LaTeX tables are
also written for each trade-off figure. No cross-dataset pooling is performed.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import recourse_statistics as rs

DATASET_LABELS = {'Synthetic':'Synthetic', 'COMPAS':'COMPAS', 'German':'German [target audit]',
    'Polish':'Polish [target audit]', 'Folktables':'FolkTables ACS Income',
    'Give_Me_Some_Credit':'Give Me Some Credit'}
CI_NOTE = 'Pointwise 95% CI across seed means; <5 seeds exploratory.\nUnavailable intervals omitted.'
METHODS = ['ORPM','ADV','ElliCE','ROAR']


def safe(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_')


def dataset_key(value):
    aliases={'folktables':'Folktables','givemesomecredit':'Give_Me_Some_Credit',
             'germancredit':'German','polishbankruptcy':'Polish'}
    return aliases.get(str(value).lower().replace(' ','').replace('_',''), str(value))


def dataset_label(value):
    return DATASET_LABELS.get(dataset_key(value),str(value))


def report_path(root, dataset, kind, name, extension):
    if kind not in {'figures','table'}:
        raise ValueError('Report kind must be figures or table')
    ds=safe(dataset_key(dataset))
    directory=Path(root)/kind/ds
    directory.mkdir(parents=True,exist_ok=True)
    return directory/f'{ds}_{safe(name)}.{extension}'


def save_table(frame, root, dataset, name, latex=False):
    frame=frame.copy()
    if not isinstance(frame.index,pd.RangeIndex) and 'Config' not in frame and frame.index.name != 'Dataset':
        frame=frame.rename_axis('Row').reset_index()
    path=report_path(root,dataset,'table',name,'csv')
    tmp=path.with_suffix('.csv.tmp')
    frame.to_csv(tmp,index=False)
    tmp.replace(path)
    if latex:
        report_path(root,dataset,'table',name,'tex').write_text(
            frame.to_latex(index=False,escape=True,float_format=lambda v:f'{v:.4g}'))
    return str(path)


def save_dataset_tables(frame,root,name):
    if frame is None or frame.empty:
        return []
    if 'Dataset' not in frame:
        raise ValueError('Dataset column is required for per-dataset exports')
    return [save_table(group.reset_index(drop=True),root,dataset,name)
            for dataset,group in frame.groupby('Dataset',sort=False)]


def save_figure(fig,root,dataset,name,show=True):
    paths=[]
    for extension in ['png','pdf']:
        path=report_path(root,dataset,'figures',name,extension)
        fig.savefig(path,dpi=250,bbox_inches='tight')
        paths.append(str(path))
    if show:
        plt.show()
    plt.close(fig)
    return paths


def _config(row):
    keys=['Num Linear','Num NN1','Num NN2']
    if all(k in row and pd.notna(row[k]) for k in keys):
        return tuple(int(row[k]) for k in keys)
    match=re.search(r'(\d+)LR\+(\d+)NN1\+(\d+)NN2',str(row.get('Config','')))
    if match is None:
        raise ValueError('Expected model composition columns or Config with LR/NN1/NN2 counts')
    return tuple(map(int,match.groups()))


def config_name(counts):
    return f'{counts[0]}LR_{counts[1]}NN1_{counts[2]}NN2'


def config_label(counts):
    return f'{sum(counts)} models: {counts[0]} LR + {counts[1]} one-layer NN + {counts[2]} two-layer NN'


def tradeoff_data(results,metric='Validity'):
    """Unpivot one summary per dataset/composition/budget, preserving seed CIs."""
    if results is None or results.empty:
        return pd.DataFrame()
    rows=[]
    unit=' (s)' if metric=='Time' else ''
    for _,row in results.iterrows():
        counts=_config(row)
        requested=sum(counts)
        if 'Requested_Models' in row and int(row['Requested_Models'])!=requested:
            raise ValueError('Requested model count disagrees with composition')
        for method in METHODS:
            prefix=f'{method} {metric}'
            if prefix+' Mean'+unit not in row:
                continue
            record={'Dataset':dataset_key(row['Dataset']),'Config':config_name(counts),
                'Num Linear':counts[0],'Num NN1':counts[1],'Num NN2':counts[2],
                'Competing Models':requested,'Actual Models':row.get('Actual Models',row.get('Total_Models',np.nan)),
                'Budget':float(row['Budget']),'Method':method,'Metric':metric,
                'Mean':row[prefix+' Mean'+unit],
                'CI95 Low':row[prefix+' CI95 Low'+unit],
                'CI95 High':row[prefix+' CI95 High'+unit],
                'N Seeds':row.get(prefix+' N Seeds',np.nan),
                'CI Status':row.get(prefix+' CI Status','unknown'),
                'Evaluated Instances':row.get(f'{method} Evaluated Instances',np.nan),
                'Run Directory':row.get('Run Directory','')}
            rows.append(record)
    data=pd.DataFrame(rows)
    if not data.empty and data.duplicated(['Dataset','Config','Budget','Method']).any():
        raise ValueError('Duplicate summaries: choose one experiment session before plotting')
    return data


def _plot_lines(data,xcol,xlabel,ylabel,title,log_x=False,validity=True):
    fig,ax=plt.subplots(figsize=(7.6,5.4))
    colors={m:plt.get_cmap('tab10')(i) for i,m in enumerate(METHODS)}
    plotted=False
    for i,(method,group) in enumerate(data.groupby('Method',sort=False)):
        sub=group.sort_values(xcol)
        if sub[xcol].duplicated().any():
            raise ValueError(f'Multiple configurations at the same {xcol} for {method}')
        x=sub[xcol].to_numpy(float); y=sub['Mean'].to_numpy(float)
        if not np.isfinite(y).any():
            continue
        color=colors.get(method,plt.get_cmap('tab10')(i%10))
        ax.plot(x,y,'o-',label=method,color=color,linewidth=1.8,markersize=5)
        low=sub['CI95 Low'].to_numpy(float);high=sub['CI95 High'].to_numpy(float)
        keep=np.isfinite(low)&np.isfinite(high)&np.isfinite(y)
        ax.fill_between(x,low,high,where=keep,color=color,alpha=.13)
        # Error bars also show intervals when only one grid point is available.
        if keep.any():
            ax.errorbar(x[keep],y[keep],yerr=np.maximum(0,np.stack([y[keep]-low[keep],high[keep]-y[keep]])),
                        fmt='none',ecolor=color,alpha=.5,capsize=3)
        plotted=True
    ticks=sorted(data[xcol].dropna().unique())
    if log_x and ticks and min(ticks)>0:
        ax.set_xscale('log')
    ax.set_xticks(ticks);ax.set_xticklabels([f'{x:g}' for x in ticks]);ax.minorticks_off()
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    if validity:
        ax.set_ylim(-2,102)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.22)
    if plotted:
        ax.legend(loc='best',fontsize=9)
    else:
        ax.text(.5,.5,'No evaluated instances',ha='center',transform=ax.transAxes)
    fig.text(.5,.015,CI_NOTE,ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.07,1,1))
    return fig


def _compact_table(data,xcol):
    result=data[[xcol,'Method','Mean','CI95 Low','CI95 High','N Seeds','CI Status']].copy()
    result['Mean [95% CI]']=result.apply(lambda r:rs.format_ci(r['Mean'],r['CI95 Low'],r['CI95 High'],r['CI Status']),axis=1)
    return result[[xcol,'Method','Mean [95% CI]','N Seeds']].sort_values([xcol,'Method'])


def plot_tradeoffs(results,root,show=True,metric='Validity',views=('budget','size')):
    """One figure per dataset/composition, and one per dataset/budget.

    The x-axis uses requested composition size, including failed configurations;
    missing evaluations remain NaN, never a spurious zero or a zero-sized set.
    """
    if set(views)-{'budget','size'}:
        raise ValueError('views must contain budget and/or size')
    data=tradeoff_data(results,metric)
    if data.empty:
        return []
    # Filtered/runtime re-plots must not replace the complete validity summary.
    if metric=='Validity' and set(views)=={'budget','size'}:
        save_dataset_tables(results,root,'cost_validity_summary')
        save_dataset_tables(data,root,'cost_validity_validity_plot_data')
    paths=[]
    ylabel='Valid recourse across all competing models (%)' if metric=='Validity' else f'Mean {metric.lower()}'
    if metric=='Time':ylabel='Mean recourse runtime (seconds)'
    for dataset,ds in data.groupby('Dataset',sort=False):
        groups=[]
        if 'budget' in views:
            for config,group in ds.groupby('Config',sort=False):
                counts=tuple(int(group[k].iloc[0]) for k in ['Num Linear','Num NN1','Num NN2'])
                groups.append((group,'Budget','Recourse budget (standardized L2 distance; log scale)',
                    f'{dataset_label(dataset)}\n{config_label(counts)}',f'{metric.lower()}_vs_budget_{config}',True))
        if 'size' in views:
            for budget,group in ds.groupby('Budget',sort=True):
                # Equal proportions in the requested sweep; retain configuration columns in CSV.
                groups.append((group,'Competing Models','Number of competing models',
                    f'{dataset_label(dataset)} — budget {budget:g}',
                    f'{metric.lower()}_vs_competing_set_size_budget_{budget:g}',False))
        for group,xcol,xlabel,title,name,log_x in groups:
            save_table(group.reset_index(drop=True),root,dataset,name+'_data')
            save_table(_compact_table(group,xcol).reset_index(drop=True),root,dataset,name,latex=True)
            fig=_plot_lines(group,xcol,xlabel,ylabel,title,log_x,metric=='Validity')
            paths.extend(save_figure(fig,root,dataset,name,show))
    return paths


def plot_convergence(entries,root,show=True,individual=False,num_instances=5):
    paths=[]
    for entry in entries:
        dataset=entry['dataset'];name=safe(entry['config'])+f"_budget_{entry['budget']:g}"
        fig,ax=plt.subplots(figsize=(7.6,5.4))
        if individual:
            records=[]
            for i,curve in enumerate(entry.get('individual_trajectories',[])[:num_instances]):
                x=np.arange(1,len(curve)+1)
                seed=entry.get('trajectory_seeds',[None]*num_instances)[i]
                ax.plot(x,curve,label=f'Trajectory {i+1} (seed {seed})',alpha=.8)
                records.extend({'Trajectory':i+1,'Seed':seed,'Round':j,'Maximum Loss':v} for j,v in zip(x,curve))
            table=pd.DataFrame(records);name='individual_convergence_'+name
            if records:ax.legend(fontsize=8)
        else:
            mean=np.asarray(entry['mean'],float);x=np.arange(1,len(mean)+1)
            ax.plot(x,mean,color='#2368a2')
            ax.fill_between(x,entry['ci_low'],entry['ci_high'],alpha=.17,color='#2368a2')
            table=pd.DataFrame({'Round':x,'Mean Maximum Loss':mean,
                'CI95 Low':entry['ci_low'],'CI95 High':entry['ci_high'],
                'N Seeds':entry['n_seeds'],'CI Status':entry['ci_status']})
            name='mean_convergence_'+name
            fig.text(.5,.015,CI_NOTE,ha='center',fontsize=8,color='#555555')
        ax.set(xlabel='ORPM round',ylabel='Maximum recourse loss (probability MSE)',ylim=(0,1),
               title=f"{dataset_label(dataset)}\n{entry['config']}; budget {entry['budget']:g}")
        ax.grid(alpha=.22);fig.tight_layout(rect=(0,.07,1,1))
        save_table(table,root,dataset,name)
        paths.extend(save_figure(fig,root,dataset,name,show))
    return paths
