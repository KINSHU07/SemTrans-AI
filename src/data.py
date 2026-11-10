"""Step 1-2: find CBIS-DDSM files, build the manifest, preprocess to a working-resolution cache, split by patient."""
import os, re, glob, json, hashlib
import numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GroupShuffleSplit

try:
    import pydicom
except ImportError:          # allows importing this module without pydicom (e.g. unit tests)
    pydicom = None


# ============================================================ discovery
def find_case_csvs(root):
    """Return {('mass'|'calc', 'train'|'test'): path} for the CBIS-DDSM case-description CSVs under root."""
    out = {}
    for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True):
        m = re.match(r'(mass|calc)_case_description_(train|test)_set', os.path.basename(p).lower())
        if m:
            out[(m.group(1), m.group(2))] = p
    return out


def _natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(path))]


IMG_EXT = ('.dcm', '.jpg', '.jpeg', '.png')


def _read_header(p):
    if not p.lower().endswith('.dcm'):                     # JPEG/PNG copies (e.g. the Kaggle version): jpeg/<SeriesUID>/1-1.jpg
        return dict(path=p, series_uid=os.path.basename(os.path.dirname(p)), patient_id='', desc='', rows=0, cols=0)
    try:
        d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        return dict(path=p, series_uid=str(getattr(d, 'SeriesInstanceUID', '')),
                    patient_id=str(getattr(d, 'PatientID', '')),
                    desc=str(getattr(d, 'SeriesDescription', '')),
                    rows=int(getattr(d, 'Rows', 0) or 0), cols=int(getattr(d, 'Columns', 0) or 0))
    except Exception:
        return None


def index_images(root, cache_csv, n_workers=4, refresh=False):
    """Scan every image file (.dcm / .jpg / .png) under root once and remember its series id. Cached because Drive is slow."""
    if os.path.exists(cache_csv) and not refresh:
        df = pd.read_csv(cache_csv)
        if len(df) and str(df['path'].iloc[0]).startswith(root):
            return df
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.lower().endswith(IMG_EXT)]
    with ThreadPoolExecutor(n_workers) as ex:
        rows = [r for r in ex.map(_read_header, files) if r]
    df = pd.DataFrame(rows, columns=['path', 'series_uid', 'patient_id', 'desc', 'rows', 'cols'])
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


index_dicoms = index_images          # old name kept so existing notebooks still work


def series_lookup(dcm_index):
    """series_uid -> list of file paths (naturally sorted, so 1-1.dcm comes before 1-2.dcm)."""
    out = {}
    for uid, g in dcm_index.groupby('series_uid'):
        out[uid] = sorted(g['path'].tolist(), key=_natural_key)
    return out


def resolve_csv_path(path_str, series_map):
    """CBIS CSV paths look like <Subject>/<StudyUID>/<SeriesUID>/000000.dcm . Match on the series UID."""
    if not isinstance(path_str, str):
        return None
    parts = path_str.strip().replace('\\', '/').split('/')
    if len(parts) < 2:
        return None
    files = series_map.get(parts[-2].strip())
    if not files:
        return None
    m = re.match(r'0*(\d+)\.dcm', parts[-1].strip(), re.I)
    k = int(m.group(1)) if m else 0
    return files[k] if k < len(files) else files[0]


def diagnose_matching(cfg, dcm_index):
    """Print how many CSV rows can be matched to an image file. Run this first if build_manifest drops many rows."""
    smap = series_lookup(dcm_index); csvs = find_case_csvs(cfg.cbis_root)
    raw = pd.concat([pd.read_csv(p) for p in csvs.values()], ignore_index=True)
    raw.columns = [c.strip().lower().replace(' ', '_') for c in raw.columns]
    for col in ['image_file_path', 'cropped_image_file_path', 'roi_mask_file_path']:
        if col not in raw:
            print(f'column {col!r} NOT in the CSVs; columns are: {list(raw.columns)[:20]}'); continue
        hit = raw[col].map(lambda x: resolve_csv_path(x, smap)).notna().mean()
        print(f'{col:26s}: {hit:6.1%} of {len(raw)} rows matched an image file')
    if 'image_file_path' in raw:
        print('example CSV path     :', repr(raw['image_file_path'].iloc[0]))
    print('example series ids   :', list(smap)[:3])


def build_manifest(cfg, dcm_index=None):
    """One row per annotated abnormality, with the DICOM files it refers to. Rows with missing files are dropped."""
    cfg.ensure()
    csvs = find_case_csvs(cfg.cbis_root)
    if not csvs:
        raise FileNotFoundError(f'No *_case_description_*_set.csv files found under {cfg.cbis_root}')
    frames = []
    for (typ, split), p in csvs.items():
        d = pd.read_csv(p)
        d.columns = [c.strip().lower().replace(' ', '_') for c in d.columns]
        d['lesion_type'], d['split_official'] = typ, split
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={'left_or_right_breast': 'side', 'image_view': 'view'})
    raw['label'] = raw['pathology'].astype(str).str.upper().str.startswith('MALIGNANT').astype(int)

    if dcm_index is None:
        dcm_index = index_dicoms(cfg.cbis_root, cfg.path('cache', 'dicom_index.csv'), cfg.n_workers)
    smap = series_lookup(dcm_index)
    raw['full_file'] = raw['image_file_path'].map(lambda s: resolve_csv_path(s, smap))
    raw['roi_file'] = raw['roi_mask_file_path'].map(lambda s: resolve_csv_path(s, smap))
    raw['crop_file'] = raw['cropped_image_file_path'].map(lambda s: resolve_csv_path(s, smap))
    n0 = len(raw)
    man = raw[raw['full_file'].notna() & (raw['roi_file'].notna() | raw['crop_file'].notna())].reset_index(drop=True)
    print(f'CSV rows: {n0} | rows whose DICOM files were found: {len(man)} | dropped: {n0 - len(man)}')
    keep = ['patient_id', 'side', 'view', 'lesion_type', 'split_official', 'abnormality_id', 'pathology',
            'label', 'subtlety', 'assessment', 'full_file', 'roi_file', 'crop_file']
    return man[[c for c in keep if c in man.columns]]


# ============================================================ image reading and geometry
def read_dicom(path):
    d = pydicom.dcmread(path)
    a = d.pixel_array.astype(np.float32)
    if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        a = a.max() - a
    return a


def read_image(path):
    """DICOM or ordinary image file -> 2-D float32 array."""
    if path.lower().endswith('.dcm'):
        return read_dicom(path)
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f'cannot read {path}')
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.shape[2] == 3 else a[..., 0]
    return a.astype(np.float32)


def normalize01(a):
    sub = a[::4, ::4]
    nz = sub[sub > 0]
    lo, hi = np.percentile(nz if nz.size else sub, [0.5, 99.5])
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def breast_geometry(img01):
    """Bounding box of the breast (largest bright component) and whether to mirror so the chest wall is on the left."""
    H, W = img01.shape
    f = 16
    small = cv2.resize(img01, (max(W // f, 1), max(H // f, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    v = small[small > 0.02]                               # non-background pixels
    ref = float(np.median(v)) if v.size else 1.0          # typical tissue level (robust to bright lesions / dense areas)
    bw = (small > max(0.15 * ref, 0.02)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[k, :4]
        my, mx = int(0.01 * H), int(0.01 * W)
        y0, y1 = max(0, y * f - my), min(H, (y + h) * f + my)
        x0, x1 = max(0, x * f - mx), min(W, (x + w) * f + mx)
    crop = img01[y0:y1, x0:x1]
    third = max(crop.shape[1] // 3, 1)
    flip = bool(crop[:, -third:].mean() > crop[:, :third].mean())     # tissue should be on the left
    return (y0, y1, x0, x1), flip


def warp_image(a, box, flip, out_hw):
    y0, y1, x0, x1 = box
    c = a[y0:y1, x0:x1]
    if flip:
        c = c[:, ::-1]
    return cv2.resize(np.ascontiguousarray(c), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)


def warp_mask(m, box, flip, out_hw):
    """Crop/flip/resize a binary mask; any overlap counts so tiny lesions do not vanish."""
    y0, y1, x0, x1 = box
    c = m[y0:y1, x0:x1]
    if flip:
        c = c[:, ::-1]
    r = cv2.resize(np.ascontiguousarray(c).astype(np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) > 0
    if not r.any() and c.any():                       # lesion smaller than one output pixel
        ys, xs = np.nonzero(c)
        r[min(int(ys.mean() * out_hw[0] / c.shape[0]), out_hw[0] - 1),
          min(int(xs.mean() * out_hw[1] / c.shape[1]), out_hw[1] - 1)] = True
    return r.astype(np.uint8)


def _looks_like_mask(a):
    """A mask is (almost) only black and white. Exactly binary for DICOM, nearly binary for JPEG (compression noise)."""
    s = a[::2, ::2].astype(np.float32); mx = float(s.max())
    if mx <= 0:
        return False
    black = float((s <= 0.1 * mx).mean())                 # real ROI masks are mostly black (a small lesion on a full-size image)
    return float(((s <= 0.1 * mx) | (s >= 0.9 * mx)).mean()) > 0.97 and black > 0.5


def load_mask(row, full_shape):
    """The ROI-mask and cropped-image columns are sometimes swapped in CBIS-DDSM: pick whichever file looks like a mask."""
    for key in ('roi_file', 'crop_file'):
        p = row.get(key)
        if isinstance(p, str):
            a = read_image(p)
            if _looks_like_mask(a):
                m = (a > 0.5 * a.max()).astype(np.uint8)
                if m.shape != tuple(full_shape):
                    m = cv2.resize(m, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST)
                return m
    return None


def process_image(full_file, rows, cfg):
    """Return (image uint8, union mask uint8, [abnormality info]) at working resolution for one mammogram."""
    full = read_image(full_file)
    img01 = normalize01(full)
    box, flip = breast_geometry(img01)
    hw = (cfg.img_h, cfg.img_w)
    img = (warp_image(img01, box, flip, hw) * 255).round().astype(np.uint8)
    union = np.zeros(hw, np.uint8)
    abns = []
    for r in rows:
        m = load_mask(r, full.shape)
        if m is None:
            continue
        mw = warp_mask(m, box, flip, hw)
        if not mw.any():
            continue
        union |= mw
        ys, xs = np.nonzero(mw)
        abns.append(dict(cy=float(ys.mean()), cx=float(xs.mean()), y0=int(ys.min()), y1=int(ys.max()) + 1,
                         x0=int(xs.min()), x1=int(xs.max()) + 1, area=int(mw.sum()),
                         label=int(r['label']), lesion_type=r.get('lesion_type', ''),
                         pathology=str(r.get('pathology', '')), subtlety=float(r.get('subtlety', np.nan))))
    return img, union, abns


# ============================================================ cache
def _key(full_file):
    return hashlib.md5(str(full_file).encode()).hexdigest()[:12]


def build_cache(cfg, manifest, verbose=True):
    """Preprocess every mammogram once (resumable), then assemble images.npy / masks.npy + two tables."""
    cfg.ensure()
    shard_dir = cfg.path('cache', 'shards'); os.makedirs(shard_dir, exist_ok=True)
    groups = {k: g for k, g in manifest.groupby('full_file')}
    keys = sorted(groups)
    if cfg.max_images and len(keys) > cfg.max_images:
        rng = np.random.default_rng(cfg.seed)
        keys = sorted(rng.choice(keys, cfg.max_images, replace=False).tolist())

    def work(ff):
        out = os.path.join(shard_dir, _key(ff) + '.npz')
        if os.path.exists(out):
            return ff, True
        try:
            rows = groups[ff].to_dict('records')
            img, union, abns = process_image(ff, rows, cfg)
            if not abns:
                return ff, False
            pid = rows[0]['patient_id']
            np.savez_compressed(out, img=img, mask=union, abns=np.array(json.dumps(abns)), patient_id=np.array(str(pid)))
            return ff, True
        except Exception as e:
            if verbose:
                print('skipped', ff, '->', repr(e)[:120])
            return ff, False

    with ThreadPoolExecutor(cfg.n_workers) as ex:
        results = list(ex.map(work, keys))
    ok = [ff for ff, good in results if good]
    if verbose:
        print(f'preprocessed {len(ok)} / {len(keys)} mammograms')

    imgs, masks, img_rows, abn_rows = [], [], [], []
    for i, ff in enumerate(ok):
        z = np.load(os.path.join(shard_dir, _key(ff) + '.npz'))
        imgs.append(z['img']); masks.append(z['mask'])
        abns = json.loads(str(z['abns']))
        labels = {a['label'] for a in abns}
        g = groups[ff].iloc[0]
        img_rows.append(dict(img_idx=i, patient_id=str(z['patient_id']), full_file=ff, side=g.get('side'),
                             view=g.get('view'), n_abn=len(abns), label=int(max(labels)), mixed=len(labels) > 1,
                             lesion_type='+'.join(sorted({a['lesion_type'] for a in abns}))))
        for j, a in enumerate(abns):
            abn_rows.append(dict(img_idx=i, abn_idx=j, patient_id=str(z['patient_id']), **a))
    images, masks = np.stack(imgs), np.stack(masks)
    np.save(cfg.path('cache', 'images.npy'), images); np.save(cfg.path('cache', 'masks.npy'), masks)
    images_df, abn_df = pd.DataFrame(img_rows), pd.DataFrame(abn_rows)
    images_df.to_csv(cfg.path('cache', 'images_raw.csv'), index=False)
    abn_df.to_csv(cfg.path('cache', 'abn_raw.csv'), index=False)
    return images, masks, images_df, abn_df


# ============================================================ splits
def make_splits(images_df, abn_df, cfg):
    """Patient-level train/val/test split. Both tables get a 'split' column; saved to the cache folder."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.val_frac + cfg.test_frac, random_state=cfg.seed)
    tr, rest = next(gss1.split(images_df, groups=images_df.patient_id))
    rest_df = images_df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac), random_state=cfg.seed)
    va, te = next(gss2.split(rest_df, groups=rest_df.patient_id))
    split = np.empty(len(images_df), dtype=object)
    split[tr] = 'train'; split[rest_df.index[va]] = 'val'; split[rest_df.index[te]] = 'test'
    images_df = images_df.copy(); images_df['split'] = split
    abn_df = abn_df.copy(); abn_df['split'] = abn_df['img_idx'].map(images_df.set_index('img_idx')['split'])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(images_df[images_df.split == a].patient_id) & set(images_df[images_df.split == b].patient_id)
    images_df.to_csv(cfg.path('cache', 'images.csv'), index=False)
    abn_df.to_csv(cfg.path('cache', 'abn.csv'), index=False)
    return images_df, abn_df


def grouped_kfold(images_df, k=5, seed=42):
    """Optional: patient-level k-fold assignment (adds a 'fold' column) for cross-validated final results."""
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(seed)
    pats = np.array(images_df.patient_id.unique().tolist()); rng.shuffle(pats)
    order = {p: i for i, p in enumerate(pats)}
    g = images_df.patient_id.map(order)
    fold = np.zeros(len(images_df), int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=k).split(images_df, groups=g)):
        fold[te] = f
    out = images_df.copy(); out['fold'] = fold
    return out


def load_cache(cfg, mmap=False):
    images = np.load(cfg.path('cache', 'images.npy'), mmap_mode='r' if mmap else None)
    masks = np.load(cfg.path('cache', 'masks.npy'), mmap_mode='r' if mmap else None)
    images_df = pd.read_csv(cfg.path('cache', 'images.csv'))
    abn_df = pd.read_csv(cfg.path('cache', 'abn.csv'))
    return images, masks, images_df, abn_df


def qc_summary(images_df, abn_df):
    print('images:', len(images_df), '| abnormalities:', len(abn_df), '| patients:', images_df.patient_id.nunique())
    print('mixed-label images (excluded from image-level evaluation):', int(images_df.mixed.sum()))
    print(images_df.groupby('split')['label'].agg(['count', 'mean']).rename(columns={'mean': 'malignant_frac'}).round(3))
    print(abn_df.groupby('lesion_type')['label'].agg(['count', 'mean']).round(3))
    print('lesion area at working resolution (pixels):', abn_df.area.describe()[['min', '50%', 'max']].round(0).to_dict())
