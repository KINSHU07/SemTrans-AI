import os, tempfile
import sys, numpy as np, pandas as pd
import sklearn.metrics, scipy.stats, skimage.metrics
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import viz, metrics as mt
rng=np.random.default_rng(0)
# transmission summary shaped like pipeline.summarize output
rows=[]
for rcv in ['standard','robust']:
    rows.append(dict(receiver=rcv,scheme='clean',source='none',rate=0.0,AUC=.9,AUC_sd=0,sens_at_spec=.5,loc_hit=.95,lesion_loss=0))
    for sch,src in [('uniform','none'),('burst','none'),('semantic','pred'),('semantic','oracle')]:
        for r in [.1,.3,.5]:
            rows.append(dict(receiver=rcv,scheme=sch,source=src,rate=r,AUC=.9-r*(0.6 if src=='none' else .1),AUC_sd=.02,sens_at_spec=.5,loc_hit=.9-r/2,lesion_loss=r if src=='none' else 0))
viz.plot_transmission(pd.DataFrame(rows),'standard',path=tempfile.gettempdir() + '/v1.png')
yt=np.r_[np.zeros(60),np.ones(60)]; p=np.r_[rng.normal(.3,.2,60),rng.normal(.7,.2,60)].clip(0,1)
viz.plot_risk_coverage({'a':mt.risk_coverage(yt,p,-np.abs(p-.5)),'b':mt.risk_coverage(yt,p,rng.random(120))},path=tempfile.gettempdir() + '/v2.png')
viz.plot_reliability(yt,{'raw':p,'T':p**1.2},path=tempfile.gettempdir() + '/v3.png')
recs=[dict(strategy=s,seed=sd,round=r,n_labeled=50+30*r,AUC=.7+.03*r+rng.normal(0,.01)+(.01 if s=='entropy' else 0)) for s in ['random','entropy','bald','coreset'] for sd in range(3) for r in range(6)]
viz.plot_al(pd.DataFrame(recs),path=tempfile.gettempdir() + '/v4.png',target=.85)
img=rng.random((256,192)).astype(np.float32); imp=np.zeros((8,6),np.int8); imp[3,2]=2
viz.plot_loss_example(img,img,imp,None,None,32,img,img,path=tempfile.gettempdir() + '/v5.png')
imgs=(rng.random((8,256,192))*255).astype(np.uint8); masks=np.zeros_like(imgs); masks[:,100:130,60:90]=1
idf=pd.DataFrame(dict(patient_id=[f'P{i}' for i in range(8)],label=rng.integers(0,2,8)))
viz.show_samples(imgs,masks,idf,n=8,path=tempfile.gettempdir() + '/v6.png')
print('all 6 figure functions ran')
