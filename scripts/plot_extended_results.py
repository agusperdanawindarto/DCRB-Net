from pathlib import Path
import csv,json,statistics,shutil,hashlib
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
D=R/'figures';D.mkdir(exist_ok=True)
models=['UNet_Kvasir_only','UNet_combined','DCRBNet_combined']
labels=['U-Net Kvasir-only','U-Net pooled','DCRB-Net pooled']
seeds=[42,123,2025]
colors=['#526b82','#bb8647','#168779']
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ('png','svg'):fig.savefig(D/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)
def run(m,s):return R/'results/runs'/f'{m}_seed{s}'
def history(m,s):
    return [json.loads(x) for x in (run(m,s)/'history.jsonl').read_text().splitlines() if x.strip()]
per=rows(R/'results/per_seed.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in zip(axes,['external_mean_dice','external_mean_iou'],['External Dice','External IoU']):
    for m,label,c in zip(models,labels,colors):
        values=[float(next(x[key] for x in per if x['model']==m and int(x['seed'])==s))*100 for s in seeds]
        ax.plot(range(3),values,'o-',label=label,color=c)
    ax.set_xticks(range(3),seeds);ax.set_xlabel('Training seed');ax.set_ylabel('Score (%)');ax.set_title(title);ax.grid(alpha=.2);ax.set_ylim(0,60)
axes[0].legend(fontsize=8,loc='lower right');save(fig,'etis_per_seed')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
for ax,key,title in zip(axes,['val_dice','val_iou'],['Validation Dice','Validation IoU']):
    for m,label,c in zip(models[1:],labels[1:],colors[1:]):
        hs=[history(m,s) for s in seeds]
        epochs=[x['epoch'] for x in hs[0]]
        assert all([x['epoch'] for x in h]==epochs for h in hs)
        vals=[[float(x[key])*100 for x in h] for h in hs]
        mean=[statistics.mean(x) for x in zip(*vals)]
        sd=[statistics.stdev(x) for x in zip(*vals)]
        ax.plot(epochs,mean,color=c,label=label)
        ax.fill_between(epochs,[a-b for a,b in zip(mean,sd)],[a+b for a,b in zip(mean,sd)],color=c,alpha=.15)
    ax.set_title(title+' (matched pooled validation)');ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.set_ylim(0,100)
axes[0].set_ylabel('Score (%)');axes[0].legend(fontsize=8,loc='lower right');save(fig,'pooled_validation_curves')
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for ax,m,label,c in zip(axes,models,labels,colors):
    for s in seeds:
        h=history(m,s);ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=str(s))
    ax.set_title(label);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0].set_ylabel('Training objective');axes[-1].legend(title='Seed',fontsize=8);save(fig,'training_loss_by_model')
scores={}
for m in models:
    for s in seeds:
        data=rows(run(m,s)/'external_test_per_image.csv')
        scores[m,s]={v['sample_id']:{k:float(v[k]) for k in ('dice','iou')} for v in data}
ids=sorted(scores[models[0],42])
assert len(ids)==196 and all(set(x)==set(ids) for x in scores.values())
avg={m:{i:{k:statistics.mean(scores[m,s][i][k] for s in seeds) for k in ('dice','iou')} for i in ids} for m in models}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,k in zip(axes,['dice','iou']):
    b=ax.boxplot([[avg[m][i][k]*100 for i in ids] for m in models],tick_labels=['Kvasir-only','Pooled U-Net','DCRB-Net'],patch_artist=True,showmeans=True)
    for patch,c in zip(b['boxes'],colors):patch.set_facecolor(c);patch.set_alpha(.55)
    ax.set_title('ETIS '+k.upper());ax.set_ylabel('Per-record score (%)');ax.set_ylim(0,103);ax.grid(axis='y',alpha=.2)
save(fig,'etis_per_image_distribution')
deltas={k:[(avg[models[2]][i][k]-avg[models[1]][i][k])*100 for i in ids] for k in ('dice','iou')}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,k in zip(axes,['dice','iou']):
    ax.hist(deltas[k],bins=20,color=colors[2],edgecolor='white')
    ax.axvline(0,color='#333333',ls='--')
    ax.set_xlabel('DCRB-Net minus pooled U-Net (percentage points)');ax.set_ylabel('ETIS records');ax.set_title('Paired '+k.upper()+' differences')
save(fig,'etis_paired_differences')
hashes={x['sample_id']:x['image_sha256'] for x in rows(R/'data_audit/source_sha256_manifest.csv') if x['dataset']=='ETIS-Larib'}
assert set(hashes)==set(ids) and len(set(hashes.values()))==193
groups=defaultdict(list)
for i in ids:groups[hashes[i]].append(i)
# Average annotation scores per identical image hash; retain original 196-record primary evaluation.
sensitivity=[]
for m in models:
    for s in seeds:
        out={'model':m,'seed':s,'records':196,'unique_hashes':193}
        for k in ('dice','iou'):
            out[k+'_196']=statistics.mean(scores[m,s][i][k] for i in ids)
            out[k+'_193_grouped']=statistics.mean(statistics.mean(scores[m,s][i][k] for i in g) for g in groups.values())
        sensitivity.append(out)
def write_csv(p,data):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write_csv(R/'results/etis_duplicate_sensitivity.csv',sensitivity)
summary={}
for m in models:
    group=[x for x in sensitivity if x['model']==m]
    summary[m]={k:statistics.mean(x[k] for x in group) for k in ('dice_196','dice_193_grouped','iou_196','iou_193_grouped')}
paired={k:{'mean_difference_pp':statistics.mean(v),'improved_records':sum(x>1e-9 for x in v),'worse_records':sum(x<-1e-9 for x in v),'tied_records':sum(abs(x)<=1e-9 for x in v)} for k,v in deltas.items()}
(R/'results/etis_descriptive_summary.json').write_text(json.dumps({'duplicate_sensitivity':summary,'paired_differences':paired,'interpretation':'Descriptive only; no inferential significance or patient independence claimed.'},indent=2)+'\n',encoding='utf-8')
print('FIGURES GENERATED',len(list(D.glob('*.png'))),'PAIRED',json.dumps(paired),'SENSITIVITY',json.dumps(summary))
