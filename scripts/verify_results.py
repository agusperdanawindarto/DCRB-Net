from pathlib import Path
import csv,json,hashlib,statistics
R=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return list(csv.DictReader(p.open(newline='',encoding='utf-8')))
index=json.loads((R/'results/run_index.json').read_text())
assert len(index)==9
all_metrics=[]
ids=None
for x in index:
    run=x['run'];p=R/'results/runs'/run
    a=json.loads((p/'ARTIFACT_SHA256.json').read_text())
    for name in ['run_config.json','run_metrics.json','history.jsonl','full_sha256_preflight.json','validation_per_image.csv','external_test_per_image.csv']:
        assert sha(p/name)==a[name],(run,name)
    assert sha(R/'checkpoints'/run/'checkpoint_best.pt')==x['best_checkpoint_sha256']==a['checkpoint_best.pt']
    assert sha(R/x['notebook'])==x['notebook_sha256']
    m=json.loads((p/'run_metrics.json').read_text())
    for part,prefix in [('validation','validation'),('external_test','external')]:
        data=rows(p/(part+'_per_image.csv'))
        assert len(data)==m[part+'_images']
        if part=='external_test':
            assert all(v['source']=='ETIS-Larib' and v['split']=='external_test' for v in data)
            cur=sorted(v['sample_id'] for v in data)
            if ids is None:ids=cur
            assert cur==ids
        for metric in ['dice','iou']:
            assert abs(statistics.mean(float(v[metric]) for v in data)-m[prefix+'_mean_'+metric])<1e-9
    all_metrics.append(m)
    print('PASS',run)
for a in rows(R/'results/aggregate.csv'):
    group=[x for x in all_metrics if x['run'].rsplit('_seed',1)[0]==a['model']]
    assert len(group)==3
    for k in ['validation_mean_dice','validation_mean_iou','external_mean_dice','external_mean_iou']:
        vals=[x[k] for x in group]
        assert abs(statistics.mean(vals)-float(a[k+'_mean']))<1e-12
        assert abs(statistics.stdev(vals)-float(a[k+'_sd']))<1e-12
assert len(ids)==196
print('PASS: nine runs, all published original artifact hashes, identical ETIS IDs, per-image means and aggregate sample SDs.')
