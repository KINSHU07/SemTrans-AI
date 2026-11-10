import os, tempfile
import sys, os, types, pickle, numpy as np, pandas as pd, cv2
# ---------- stub pydicom (the real one is not installable here) ----------
class DS:  # minimal dataset object
    pass
def dcmread(path, stop_before_pixels=False, force=False):
    with open(path,'rb') as f: d = pickle.load(f)
    o = DS(); o.__dict__.update({k:v for k,v in d.items() if k!='pixel_array'})
    if not stop_before_pixels: o.pixel_array = d['pixel_array']
    return o
sys.modules['pydicom'] = types.SimpleNamespace(dcmread=dcmread)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Cfg
from src import data

import tempfile; _t = tempfile.mkdtemp(); root, work = _t + '/CBIS', _t + '/work'
rng = np.random.default_rng(0)
def save(path, arr, uid, patient, desc='x', mono1=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(pixel_array=arr, SeriesInstanceUID=uid, PatientID=patient, SeriesDescription=desc, Rows=arr.shape[0], Columns=arr.shape[1],
             PhotometricInterpretation='MONOCHROME1' if mono1 else 'MONOCHROME2')
    with open(path,'wb') as f: pickle.dump(d, f)

def breast(H=1200, W=800, side='L', lesions=()):
    img = rng.normal(300, 30, (H, W)).astype(np.float32)*0
    yy, xx = np.mgrid[:H,:W]
    # semicircular breast attached to the chest wall (left edge for L, right edge for R)
    cx0 = 0 if side=='L' else W
    inside = ((xx-cx0)**2/ (0.9*W)**2 + (yy-H/2)**2/(0.45*H)**2) < 1
    img[inside] = 8000 + rng.normal(0, 300, inside.sum())
    masks = []
    for (cy,cx,r) in lesions:
        m = ((yy-cy)**2+(xx-cx)**2) < r*r; img[m] = 20000; masks.append(m.astype(np.uint16)*255)
    return img.clip(0,65535).astype(np.uint16), masks

rows=[]; uidn=[0]
def uid(): uidn[0]+=1; return f'1.3.6.1.{uidn[0]:06d}'
for pi in range(1,9):
    pid=f'P_{pi:05d}'; side = 'RIGHT' if pi%2==0 else 'LEFT'
    two = (pi==3)               # image with two abnormalities
    les = [(600,300,40)] if not two else [(400,250,35),(800,300,45)]
    img, masks = breast(side=side[0], lesions=[(cy,(cx if side=='LEFT' else 800-cx),r) for cy,cx,r in les])
    subj=f'Mass-Training_{pid}_{side}_CC'; study=uid(); s_full=uid()
    save(f'{root}/{subj}/{study}/1.000000-full mammogram images-1/1-1.dcm', (65535-img if pi==1 else img), s_full, pid, 'full mammogram images', mono1=(pi==1))
    for ai,(m,(cy,cx,r)) in enumerate(zip(masks,les)):
        s_roi=uid(); crop = img[max(cy-60,0):cy+60, :].copy()[:, :120] if False else np.full((120,120), 15000, np.uint16)+rng.integers(0,500,(120,120)).astype(np.uint16)
        # ROI-mask series holds TWO images: 1-1 = mask, 1-2 = cropped  (real CBIS quirk)
        save(f'{root}/{subj}/{study}/2.000000-ROI mask images-{ai}/1-1.dcm', m, s_roi, pid, 'ROI mask images')
        save(f'{root}/{subj}/{study}/2.000000-ROI mask images-{ai}/1-2.dcm', crop, s_roi, pid, 'ROI mask images')
        full_p=f'{subj}/{study}/{s_full}/000000.dcm'
        roi_p=f'{subj}/{study}/{s_roi}/000000.dcm'; crop_p=f'{subj}/{study}/{s_roi}/000001.dcm'
        if pi in (2,5): roi_p, crop_p = crop_p, roi_p            # swapped columns quirk
        if pi==7: full_p=f'{subj}/{study}/1.9.9.9.9/000000.dcm'   # file missing on disk -> must be dropped
        path_str = lambda p: p+'\n' if pi==4 else p                # trailing newline quirk
        rows.append({'patient_id':pid,'breast_density':2,'left or right breast':side,'image view':'CC','abnormality id':ai+1,
                     'abnormality type':'mass','mass shape':'ROUND','mass margins':'CIRCUMSCRIBED','assessment':3,
                     'pathology':['BENIGN','MALIGNANT','BENIGN_WITHOUT_CALLBACK'][pi%3],'subtlety':3,
                     'image file path':path_str(full_p),'cropped image file path':path_str(crop_p),'ROI mask file path':path_str(roi_p)})
df = pd.DataFrame(rows)
os.makedirs(root, exist_ok=True)
df.iloc[:6].to_csv(f'{root}/mass_case_description_train_set.csv', index=False)
df.iloc[6:].to_csv(f'{root}/mass_case_description_test_set.csv', index=False)

cfg = Cfg(cbis_root=root, work_dir=work, img_h=256, img_w=192, n_workers=2, seed=1, val_frac=0.25, test_frac=0.25).ensure()
idx = data.index_dicoms(root, cfg.path('cache','dicom_index.csv'), 2)
print('dicoms indexed:', len(idx))
man = data.build_manifest(cfg, idx)
print(man[['patient_id','side','label','full_file']].assign(full_file=lambda d: d.full_file.str[-40:]).to_string())
images, masks, idf, adf = data.build_cache(cfg, man)
idf, adf = data.make_splits(idf, adf, cfg)
data.qc_summary(idf, adf)
print(images.shape, masks.shape, images.dtype)
# alignment check: image should be bright at lesion centroid; chest wall (tissue) should be on the left after standardisation
ok = 0
for r in adf.itertuples():
    v = images[r.img_idx][int(r.cy), int(r.cx)]; bg = images[r.img_idx].mean()
    ok += v > 1.5*bg
print('lesion centroid brighter than image mean in', ok, 'of', len(adf), 'abnormalities')
left = np.mean([images[i][:, :64].mean() for i in range(len(images))]); right = np.mean([images[i][:, -64:].mean() for i in range(len(images))])
print('mean brightness left strip %.1f vs right strip %.1f  (left should be larger)' % (left, right))
print('mask pixels per image:', masks.reshape(len(masks),-1).sum(1).tolist())
# reload path + k-fold
im2, mk2, i2, a2 = data.load_cache(cfg); assert (im2==images).all()
print(data.grouped_kfold(i2, k=2).fold.tolist())
