"""Mock of the Kaggle JPEG version of CBIS-DDSM: csv/*.csv + jpeg/<SeriesUID>/1-1.jpg (real JPEG compression noise)."""
import os, sys, tempfile, numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Cfg
from src import data

tmp = tempfile.mkdtemp(); root = tmp + '/kaggle'; work = tmp + '/work'
os.makedirs(root + '/csv'); rng = np.random.default_rng(0)
yy, xx = np.mgrid[:1200, :800]

def make(side, lesions):
    img = np.zeros((1200, 800), np.float32); cx0 = 0 if side == 'L' else 800
    inside = ((xx - cx0) ** 2 / (0.9 * 800) ** 2 + (yy - 600) ** 2 / (0.45 * 1200) ** 2) < 1
    img[inside] = 0.35 + rng.normal(0, 0.02, inside.sum()); masks = []
    for (cy, cx, r) in lesions:
        cx = cx if side == 'L' else 800 - cx; m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r; img[m] = 0.9; masks.append(m)
    return (img.clip(0, 1) * 255).astype(np.uint8), masks

def jpg(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True); cv2.imwrite(path, a, [cv2.IMWRITE_JPEG_QUALITY, 90])

rows, n = [], [0]
def uid(): n[0] += 1; return f'1.3.6.1.4.1.9590.100.1.2.{n[0]:08d}'
for pi in range(1, 9):
    pid = f'P_{pi:05d}'; side = 'RIGHT' if pi % 2 == 0 else 'LEFT'
    les = [(600, 300, 40)] if pi != 3 else [(400, 250, 35), (800, 300, 45)]
    img, masks = make(side[0], les); subj = f'Mass-Training_{pid}_{side}_CC'; study, s_full = uid(), uid()
    jpg(f'{root}/jpeg/{s_full}/1-1.jpg', img)
    for ai, m in enumerate(masks):
        s_roi = uid(); jpg(f'{root}/jpeg/{s_roi}/1-1.jpg', m.astype(np.uint8) * 255)            # mask  (1-1)
        jpg(f'{root}/jpeg/{s_roi}/1-2.jpg', (120 + rng.integers(0, 60, (120, 120))).astype(np.uint8))   # crop (1-2), same series
        roi_p, crop_p = f'{subj}/{study}/{s_roi}/000000.dcm', f'{subj}/{study}/{s_roi}/000001.dcm'
        if pi in (2, 5): roi_p, crop_p = crop_p, roi_p                                             # swapped-column quirk
        rows.append({'patient_id': pid, 'breast_density': 2, 'left or right breast': side, 'image view': 'CC', 'abnormality id': ai + 1,
                     'abnormality type': 'mass', 'assessment': 3, 'pathology': ['BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK'][pi % 3],
                     'subtlety': 3, 'image file path': f'{subj}/{study}/{s_full}/000000.dcm',
                     'cropped image file path': crop_p, 'ROI mask file path': roi_p})
df = pd.DataFrame(rows)
df.iloc[:6].to_csv(root + '/csv/mass_case_description_train_set.csv', index=False)
df.iloc[6:].to_csv(root + '/csv/mass_case_description_test_set.csv', index=False)
pd.DataFrame({'SeriesDescription': ['x']}).to_csv(root + '/csv/dicom_info.csv', index=False)      # present in the real dataset, not needed

cfg = Cfg(cbis_root=root, work_dir=work, img_h=256, img_w=192, n_workers=2, seed=1, val_frac=0.25, test_frac=0.25).ensure()
idx = data.index_images(root, cfg.path('cache', 'img_index.csv'), 2)
print('image files indexed:', len(idx), '| series folders:', idx.series_uid.nunique())
data.diagnose_matching(cfg, idx)
man = data.build_manifest(cfg, idx)
images, masks, idf, adf = data.build_cache(cfg, man)
idf, adf = data.make_splits(idf, adf, cfg); data.qc_summary(idf, adf)
areas = masks.reshape(len(masks), -1).sum(1)
print('mask pixels per image (expect ~250-400, two lesions ~700):', areas.tolist())
ok = sum(images[r.img_idx][int(r.cy), int(r.cx)] > 1.5 * images[r.img_idx].mean() for r in adf.itertuples())
print(f'lesion centroid brighter than image mean in {ok} of {len(adf)} abnormalities')
left = np.mean([im[:, :64].mean() for im in images]); right = np.mean([im[:, -64:].mean() for im in images])
print(f'left strip {left:.1f} vs right strip {right:.1f} (left should be larger)')
assert len(adf) == 9 and ((areas > 150) & (areas < 1200)).all() and ok == len(adf) and left > right
print('KAGGLE-JPEG ROUTE TEST PASSED')
