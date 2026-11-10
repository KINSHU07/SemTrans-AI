import os, tempfile
import sys, types, numpy as np, pandas as pd, cv2
from unittest.mock import MagicMock
import sklearn.metrics, sklearn.model_selection, scipy.stats, skimage.metrics   # import before stubbing torch
# stub torch stack so the pure-logic parts of pipeline.py can be imported and tested
for name in ['torch','torch.nn','torch.nn.functional','torch.utils','torch.utils.data','torchvision','torchvision.models']:
    sys.modules[name] = MagicMock()
class _Base: pass
sys.modules['torch.utils.data'].Dataset = _Base
sys.modules['torch.nn'].Module = _Base
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Cfg
from src import channel as ch, metrics as mt
import src.pipeline as pl

cfg = Cfg(img_h=256, img_w=192, block=32, reps=2, loss_rates=(0.0,0.2,0.4,0.6), seed=3, mc_passes=5, patch=96, patch_out=96)
rng = np.random.default_rng(0)
# ---------- synthetic test set: lesion brightness decides the label ----------
N=60; imgs=[]; masks=[]; y=[]
yy,xx=np.mgrid[:256,:192]
for i in range(N):
    img=np.zeros((256,192),np.float32); br=((yy-128)**2/115**2+(xx)**2/170**2)<1; img[br]=0.40+rng.normal(0,0.02,br.sum())
    cy,cx=rng.integers(70,190),rng.integers(30,120); les=((yy-cy)**2+(xx-cx)**2)<14**2
    lab=int(i%2==0); img[les]=0.95 if lab else 0.70; imgs.append((img.clip(0,1)*255).astype(np.uint8)); masks.append(les.astype(np.uint8)); y.append(lab)
imgs=np.stack(imgs); masks=np.stack(masks); y=np.array(y)
meta=pd.DataFrame(dict(patient_id=[f'P{i//2}' for i in range(N)], label=y))

# ---------- channel unit checks ----------
img01=imgs[0]/255.; tx=ch.tx_block_mask(img01,32); print('tx blocks',tx.sum(),'of',tx.size)
imp=ch.block_importance(masks[0].astype(np.float32),tx,32,near=1); print('importance counts', {v:int((imp==v).sum()) for v in (-1,0,1,2)})
for scheme in ['uniform','burst','semantic']:
    lost=ch.choose_lost(imp,0.3,scheme,np.random.default_rng(1)); 
    print(scheme,'lost',int(lost.sum()),'expected',round(0.3*tx.sum()),'lesion blocks lost',int((lost&(imp==2)).sum()),'lost outside tx',int((lost&~tx).sum()))
r1=ch.choose_lost(imp,0.5,'burst',np.random.default_rng(2)); runs=np.diff(np.flatnonzero(np.r_[0,r1.ravel()[tx.ravel()],0]!=0)) 
for m in ['mean','zero','inpaint']:
    out,pix=ch.apply_loss(img01,ch.choose_lost(imp,0.3,'uniform',np.random.default_rng(1)),32,m); print(m,'changed px',int(pix.sum()),'finite',bool(np.isfinite(out).all()))
lost=ch.choose_lost(imp,0.5,'uniform',np.random.default_rng(1)); req=ch.region_blocks(100,100,48+32,imp.shape,32)
nl,n=ch.retransmit(lost,req,0.5,np.random.default_rng(1)); print('retx asked',n,'recovered',int(lost.sum()-nl.sum()))

# ---------- metrics ----------
yt=np.array([0]*50+[1]*50); p=np.r_[np.random.default_rng(0).normal(0.3,0.15,50),np.random.default_rng(1).normal(0.7,0.15,50)].clip(0,1)
print('AUC',round(mt.auc_safe(yt,p),3),'sens@90spec',round(mt.sens_at_spec(yt,p),3),'ECE',round(mt.ece(yt,p),3),'Brier',round(mt.brier(yt,p),3))
g=np.repeat(np.arange(50),2); print('boot CI',[round(v,3) for v in mt.bootstrap_ci(mt.auc_safe,yt,p,g,n=200)])
print('paired diff',[round(v,3) for v in mt.paired_bootstrap_diff(mt.auc_safe,yt,p,p[::-1],g,n=200)])
T=mt.fit_temperature(mt.logit(p),yt); print('temperature',round(T,2)); print('risk-coverage',[(r['coverage'],round(r['accuracy'],2)) for r in mt.risk_coverage(yt,p,np.abs(p-0.5)*-1)][:3])
print('AURC',round(mt.aurc(yt,p,-np.abs(p-0.5)),3),'err-detect AUROC',round(mt.error_detection_auroc(yt,p,-np.abs(p-0.5)),3))

# ---------- fake receiver: bright blob = lesion ----------
class FakeReceiver:
    def diagnose(self, im):
        cy,cx,pp,ent,loc=[],[],[],[],[]
        for a in im:
            b=cv2.GaussianBlur(a,(0,0),3); k=int(np.argmax(b)); y0,x0=divmod(k,b.shape[1]); m=float(b.max()); p=float(np.clip((m-0.5)/0.4,0,1))
            cy.append(y0); cx.append(x0); pp.append(p); ent.append(float(-(p*np.log(p+1e-6)+(1-p)*np.log(1-p+1e-6)))); loc.append(m)
        pp=np.array(pp); return dict(cy=np.array(cy),cx=np.array(cx),p=pp,ent=np.array(ent),mi=np.zeros(len(pp)),loc_conf=np.array(loc),logit=np.zeros(len(pp)))
heat=masks.astype(np.float16)
imps={s:pl.build_importance(cfg,imgs,masks,heat,s) for s in ['none','pred','oracle']}
conds=[('uniform','none'),('burst','none'),('semantic','pred'),('semantic','oracle')]
df=pl.run_transmission(cfg,imgs,masks,meta,{'fake':FakeReceiver()},imps,conds,verbose=False)
sm=pl.summarize(df); print(sm[['scheme','source','rate','AUC','loc_hit','lesion_loss','psnr_roi']].round(3).to_string())
bt=pl.bootstrap_table(df,'fake',0.4,[(('semantic','pred'),('uniform','none'))],n=200); print(bt.round(3).to_string())

# ---------- adaptive loop ----------
img01s=np.stack([a/255. for a in imgs]).astype(np.float32)
thr=0.3
for scheme,src in [('uniform','none'),('semantic','pred')]:
    res=pl.adaptive_loop(cfg,FakeReceiver(),img01s,imps[src],scheme,0.4,thr,seed=0); print(scheme,{k:round(v,3) for k,v in pl.adaptive_summary(y,res).items()})
