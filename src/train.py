"""Datasets, training loops, prediction and uncertainty (MC dropout, deep ensembles)."""
import os, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .models import UNetR18, build_classifier
from .channel import tx_block_mask, uniform_importance, choose_lost, apply_loss
from .metrics import auc_safe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN3 = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD3 = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def to01(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def _scaler():
    enabled = DEVICE.type == 'cuda'
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast():
    return torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda')


# ============================================================ patches
def extract_patch(img, cy, cx, size, out):
    """Square window of side `size` centred on (cy, cx) (zero padded at the border), resized to out x out."""
    half = size // 2; H, W = img.shape
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    pad = np.zeros((size, size), np.float32)
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(H, y0 + size), min(W, x0 + size)
    if ye > ys and xe > xs:
        pad[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return cv2.resize(pad, (out, out), interpolation=cv2.INTER_AREA if size > out else cv2.INTER_LINEAR)


def to_input(p):
    return ((np.repeat(p[None], 3, axis=0) - MEAN3) / STD3).astype(np.float32)


def batch_to_tensor(patches):
    x = np.repeat(np.asarray(patches, np.float32)[:, None], 3, axis=1)
    return torch.from_numpy(((x - MEAN3[None]) / STD3[None]).astype(np.float32))


def augment_loss(img01, cfg, rng):
    """Robust-training augmentation: simulate a random packet loss (uniform or bursty, random rate)."""
    rate = rng.uniform(0, cfg.loss_aug_max)
    imp = uniform_importance(tx_block_mask(img01, cfg.block))
    lost = choose_lost(imp, rate, 'uniform' if rng.random() < 0.5 else 'burst', rng, cfg.burst_len)
    return apply_loss(img01, lost, cfg.block, 'mean')[0]


def _worker_rng(ds):
    if ds.rng is None:
        info = torch.utils.data.get_worker_info()
        ds.rng = np.random.default_rng((info.seed if info else torch.initial_seed()) % (2 ** 32))
    return ds.rng


class PatchDS(Dataset):
    """Lesion patches cut on the fly from the working-resolution mammograms (centre jitter, flips, optional packet-loss)."""

    def __init__(self, images, abn, cfg, train=True, loss_aug=False):
        self.images, self.abn, self.cfg, self.train, self.loss_aug = images, abn.reset_index(drop=True), cfg, train, loss_aug
        self.rng = None

    def __len__(self):
        return len(self.abn)

    def __getitem__(self, i):
        r = self.abn.iloc[i]; rng = _worker_rng(self); c = self.cfg
        img = to01(self.images[int(r.img_idx)]); cy, cx = float(r.cy), float(r.cx)
        if self.train:
            if self.loss_aug and rng.random() < 0.7:
                img = augment_loss(img, c, rng)
            j = 0.25 * c.patch
            cy += rng.uniform(-j, j); cx += rng.uniform(-j, j)
        p = extract_patch(img, cy, cx, c.patch, c.patch_out)
        if self.train:
            if rng.random() < 0.5: p = p[:, ::-1]
            if rng.random() < 0.5: p = p[::-1, :]
            p = np.clip(p * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(to_input(np.ascontiguousarray(p))), torch.tensor(float(r.label))


class ArrDS(Dataset):
    """Pre-extracted fixed patches (used by the active-learning experiments)."""

    def __init__(self, patches, y, train=True):
        self.patches, self.y, self.train = patches, np.asarray(y), train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        p = self.patches[i].astype(np.float32)
        if self.train:
            if random.random() < 0.5: p = p[:, ::-1]
            if random.random() < 0.5: p = p[::-1, :]
        return torch.from_numpy(to_input(np.ascontiguousarray(p))), torch.tensor(float(self.y[i]))


# ============================================================ classifier: training and prediction
@torch.no_grad()
def classifier_probs(m, patches, bs=128):
    m.eval(); out = []
    for s in range(0, len(patches), bs):
        out.append(torch.sigmoid(m(batch_to_tensor(patches[s:s + bs]).to(DEVICE)).squeeze(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit_classifier(train_loader, val_patches, val_y, cfg, epochs, seed, pos_weight=1.0, verbose=True):
    """Fine-tune a fresh classifier; keep the epoch with the best validation AUC."""
    seed_everything(seed)
    m = build_classifier(cfg.pretrained).to(DEVICE)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.clf_lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=DEVICE))
    best, state = -1.0, None
    for ep in range(epochs):
        m.train(); tot, n = 0.0, 0
        for x, y in train_loader:
            if x.shape[0] < 2:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(m(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        auc = auc_safe(val_y, classifier_probs(m, val_patches)); score = 0.0 if np.isnan(auc) else auc
        if score >= best:
            best, state = score, {k: v.detach().clone() for k, v in m.state_dict().items()}
        if verbose:
            print(f'  epoch {ep + 1:2d}/{epochs} | train loss {tot / max(n, 1):.4f} | val AUC {auc:.4f}')
    m.load_state_dict(state); m.eval()
    return m, best


def train_patch_classifier(cfg, images, abn_train, val_patches, val_y, seed=0, robust=False, name=None, verbose=True):
    ds = PatchDS(images, abn_train, cfg, train=True, loss_aug=robust)
    dl = DataLoader(ds, batch_size=cfg.clf_bs, shuffle=True, num_workers=cfg.num_workers, drop_last=len(ds) > cfg.clf_bs)
    npos = int(abn_train.label.sum()); pw = (len(abn_train) - npos) / max(npos, 1)
    m, best = fit_classifier(dl, val_patches, val_y, cfg, cfg.clf_epochs, seed, pw, verbose)
    if name:
        torch.save(m.state_dict(), cfg.path('models', f'classifier_{name}.pt'))
    return m, best


def load_classifier(cfg, name):
    m = build_classifier(False).to(DEVICE)
    m.load_state_dict(torch.load(cfg.path('models', f'classifier_{name}.pt'), map_location=DEVICE)); m.eval()
    return m


def gt_patches(images, abn, cfg):
    """Fixed, lesion-centred patches (no jitter) for validation, testing and active learning."""
    return np.stack([extract_patch(to01(images[int(r.img_idx)]), r.cy, r.cx, cfg.patch, cfg.patch_out)
                     for r in abn.itertuples()]).astype(np.float16)


# ============================================================ uncertainty
def _H(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    return -(x * torch.log(x) + (1 - x) * torch.log(1 - x))


@torch.no_grad()
def extract_features(m, patches, bs=128):
    m.eval(); backbone = nn.Sequential(*list(m.children())[:-1])
    return torch.cat([backbone(batch_to_tensor(patches[s:s + bs]).to(DEVICE)).flatten(1) for s in range(0, len(patches), bs)])


@torch.no_grad()
def mc_predict(m, patches, T=20, bs=128, return_feats=False):
    """Monte Carlo dropout. Dropout sits only in the head, so CNN features are computed once and the head is run T times.
    Returns mean probability p, predictive entropy ent (total uncertainty), mutual information mi (model/epistemic part)."""
    feats = extract_features(m, patches, bs)
    head = m.fc; head.train()
    ps = torch.stack([torch.sigmoid(head(feats).squeeze(1)) for _ in range(T)]); head.eval()
    p = ps.mean(0).clamp(1e-6, 1 - 1e-6)
    out = dict(p=p.cpu().numpy(), ent=_H(p).cpu().numpy(), mi=(_H(p) - _H(ps).mean(0)).cpu().numpy(),
               logit=torch.log(p / (1 - p)).cpu().numpy())
    if return_feats:
        out['feats'] = feats
    return out


@torch.no_grad()
def ensemble_predict(models, patches, bs=128):
    """Deep ensemble: average the members' probabilities; disagreement between members = mutual information."""
    ps = torch.stack([torch.from_numpy(classifier_probs(m, patches, bs)) for m in models])
    p = ps.mean(0).clamp(1e-6, 1 - 1e-6)
    return dict(p=p.numpy(), ent=_H(p).numpy(), mi=(_H(p) - _H(ps).mean(0)).numpy(), logit=torch.log(p / (1 - p)).numpy())


def predict_uncertainty(models, patches, T=20):
    return mc_predict(models[0], patches, T) if len(models) == 1 else ensemble_predict(models, patches)


# ============================================================ localizer
class SegDS(Dataset):
    def __init__(self, images, masks, idxs, cfg, train=True, loss_aug=False):
        self.images, self.masks, self.idxs, self.cfg, self.train, self.loss_aug = images, masks, list(idxs), cfg, train, loss_aug
        self.rng = None

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]; rng = _worker_rng(self)
        img = to01(self.images[i]); m = self.masks[i].astype(np.float32)
        if self.train:
            if self.loss_aug and rng.random() < 0.7:
                img = augment_loss(img, self.cfg, rng)
            img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(m[None])


@torch.no_grad()
def predict_heatmaps(model, imgs, bs=8):
    model.eval(); out = []
    for s in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack([to01(a) for a in imgs[s:s + bs]])[:, None]).to(DEVICE)
        with _autocast():
            lg = model(x)
        out.append(torch.sigmoid(lg.float()).squeeze(1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def localize(prob, sigma=5.0):
    """Most likely lesion centre: argmax of the smoothed probability map. Returns (cy, cx, confidence)."""
    sm = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), sigma)
    k = int(np.argmax(sm)); cy, cx = divmod(k, sm.shape[1])
    return float(cy), float(cx), float(sm.max())


def hit_rate(heat, masks, dilate=24):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)); hits = []
    for h, m in zip(heat, masks):
        cy, cx, _ = localize(h); d = cv2.dilate(m.astype(np.uint8), k)
        hits.append(bool(d[int(round(cy)), int(round(cx))]))
    return float(np.mean(hits))


def dice_np(heat, masks, thr=0.5):
    d = []
    for h, m in zip(heat, masks):
        p = h.astype(np.float32) > thr; g = m > 0
        d.append((2 * (p & g).sum() + 1e-6) / (p.sum() + g.sum() + 1e-6))
    return float(np.mean(d))


def train_localizer(cfg, images, masks, train_idx, val_idx, robust=False, name=None, verbose=True):
    seed_everything(cfg.seed)
    m = UNetR18(cfg.pretrained).to(DEVICE)
    dl = DataLoader(SegDS(images, masks, train_idx, cfg, True, robust), batch_size=cfg.loc_bs, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=len(train_idx) > cfg.loc_bs)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.loc_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.loc_epochs, 1))
    scaler = _scaler(); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=DEVICE))
    best, state = -1.0, None
    val_imgs = [images[i] for i in val_idx]; val_masks = [masks[i] for i in val_idx]
    for ep in range(cfg.loc_epochs):
        m.train(); tot, n = 0.0, 0
        for x, y in dl:
            if x.shape[0] < 2:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            with _autocast():
                lg = m(x)
            lgf = lg.float(); p = torch.sigmoid(lgf)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = bce(lgf, y) + dice
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        sched.step()
        heat = predict_heatmaps(m, val_imgs); hr, dc = hit_rate(heat, val_masks), dice_np(heat, val_masks)
        score = hr + dc
        if score >= best:
            best, state = score, {k: v.detach().clone() for k, v in m.state_dict().items()}
        if verbose:
            print(f'  epoch {ep + 1:2d}/{cfg.loc_epochs} | loss {tot / max(n, 1):.4f} | val hit-rate {hr:.3f} | val Dice {dc:.3f}')
    m.load_state_dict(state); m.eval()
    if name:
        torch.save(m.state_dict(), cfg.path('models', f'localizer_{name}.pt'))
    return m


def load_localizer(cfg, name):
    m = UNetR18(False).to(DEVICE)
    m.load_state_dict(torch.load(cfg.path('models', f'localizer_{name}.pt'), map_location=DEVICE)); m.eval()
    return m
