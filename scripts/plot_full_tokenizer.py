"""Plot measured full-data milestones from JSON; contains no model execution."""
import argparse
import json
from pathlib import Path


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((root/'progress.json').read_text())
    points=data['points']
    x=[p['step']/1000 for p in points]
    validity=[100*p['full_panel']['valid_rate'] for p in points]
    cd=[p['paired_geometry']['candidate_mean_chamfer_common']/p['paired_geometry']['baseline_mean_chamfer_common'] for p in points]
    volume=[p['paired_volume']['candidate_median_relative_error']/p['paired_volume']['baseline_median_relative_error'] for p in points]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4),layout='constrained')
    axes[0].plot(x,validity,'o-',color='#1765a6',label='Full-data CE')
    axes[0].axhline(100*data['baseline']['valid_rate'],color='#666666',linestyle='--',label='Existing tokenizer')
    for a,b in zip(x,validity):axes[0].annotate(f'{b:.1f}%',(a,b),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set(title='Valid solid reconstruction',ylabel='Percent of kernel-valid originals',ylim=(40,100))
    axes[0].legend(loc='lower right',fontsize=8)
    for axis,values,title,color in zip(axes[1:],(cd,volume),('Normalized shape error','Relative volume error'),('#7650a2','#12806a')):
        axis.plot(x,values,'o-',color=color)
        axis.axhline(1,color='#666666',linestyle='--')
        axis.set(title=title,ylabel='Candidate / existing error (lower is better)')
    for axis in axes:
        axis.set_xlabel('Optimizer updates (thousands)')
        axis.grid(alpha=.25)
        axis.set_xlim(0,max(52.5,max(x)+2.5))
    fig.suptitle(f'Full-data CE tokenizer: {data["baseline"]["cases"]} validation shapes, {data["baseline"]["known_valid_references"]} kernel-valid originals',fontsize=12)
    fig.supxlabel('Shape uses mean squared Chamfer; volume uses median relative error. Ratios use shared valid IDs at each checkpoint; shared subsets vary. Lines connect measurements.',fontsize=8)
    fig.savefig(root/'learning_curve.png',dpi=180)
    fig.savefig(root/'learning_curve.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/tokenizer-full-ce-v1'))
    plot(parser.parse_args().root)
