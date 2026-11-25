from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
D=R/'figures';D.mkdir(exist_ok=True)
with (R/'results/aggregate.csv').open() as f: a=list(csv.DictReader(f))
labels=['U-Net\nKvasir-only','U-Net\npooled','DCRB-Net\npooled']
fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained')
for i,(k,label) in enumerate([('external_mean_dice','Dice'),('external_mean_iou','IoU')]):
    vals=[float(v[k+'_mean'])*100 for v in a];sd=[float(v[k+'_sd'])*100 for v in a]
    xs=[j+(i-.5)*.34 for j in range(3)]
    bars=ax.bar(xs,vals,.32,yerr=sd,capsize=3,label=label)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=10,fontsize=8)
ax.set_xticks(range(3),labels);ax.set_ylim(0,60);ax.set_ylabel('ETIS-Larib score (%)');ax.legend(frameon=False)
ax.set_title('External segmentation performance (mean ± SD, 3 seeds)')
for ext in ('png','svg'):fig.savefig(D/f'etis_comparison.{ext}',dpi=600)
plt.close(fig)
