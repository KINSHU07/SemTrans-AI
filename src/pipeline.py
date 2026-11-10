"""End-to-end experiments: transmission (E1), uncertainty + adaptive loop (E3), active learning (E4)."""
import numpy as np, pandas as pd, cv2
from torch.utils.data import DataLoader
from .channel import (tx_block_mask, uniform_importance, block_importance, choose_lost, apply_loss, region_blocks, retransmit)
from .metrics import (auc_safe, sens_at_spec, bootstrap_ci, paired_bootstrap_diff, roi_bbox, roi_psnr, roi_ssim,
                      area_under_curve)
from .train import (extract_patch, predict_heatmaps, localize, predict_uncertainty, mc_predict, extract_features,
                    classifier_probs, ArrDS, fit_classifier, to01, DEVICE)


# ============================================================ receiver and sender
class Receiver:
    """Receiver side: locate the most likely lesion in the (damaged) image, crop it, classify, report uncertainty."""

    def __init__(self, cfg, localizer, classifiers):
        self.cfg, self.loc, self.clf = cfg, localizer, list(classifiers)

    def diagnose(self, imgs01):
        c = self.cfg
        heat = predict_heatmaps(self.loc, imgs01)
        locs = [localize(h) for h in heat]
        patches = np.stack([extract_patch(im, cy, cx, c.patch, c.patch_out) for im, (cy, cx, _) in zip(imgs01, locs)])
        u = predict_uncertainty(self.clf, patches, c.mc_passes)
        u.update(cy=np.array([l[0] for l in locs]), cx=np.array([l[1] for l in locs]), loc_conf=np.array([l[2] for l in locs]))
        return u


def diagnose_all(receiver, imgs01, chunk=48):
    parts = [receiver.diagnose(imgs01[s:s + chunk]) for s in range(0, len(imgs01), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build_importance(cfg, images_u8, masks_u8=None, heat=None, source='pred'):
    """Sender-side importance map for every image. source: 'pred' (sender runs the localizer), 'oracle' (true mask), 'none' (no priority)."""
    out = []
    for i in range(len(images_u8)):
        tx = tx_block_mask(to01(images_u8[i]), cfg.block)
        if source == 'none':
            out.append(uniform_importance(tx))
        else:
            prob = masks_u8[i].astype(np.float32) if source == 'oracle' else heat[i].astype(np.float32)
            out.append(block_importance(prob, tx, cfg.block, near=cfg.near_blocks))
    return out


# ============================================================ E1: transmission experiment
def run_transmission(cfg, images_u8, masks_u8, meta, receivers, importance, conditions, rates=None, reps=None,
                     recon=None, chunk=48, verbose=True):
    """For every (scheme, importance source, loss rate, repeat): damage the images, reconstruct, diagnose with each receiver.
    conditions: [('uniform','none'), ('burst','none'), ('semantic','pred'), ('semantic','oracle')]
    importance: {'none': [...], 'pred': [...], 'oracle': [...]}   meta: DataFrame with patient_id, label (test images)."""
    rates = cfg.loss_rates if rates is None else rates
    reps = cfg.reps if reps is None else reps
    recon = cfg.recon if recon is None else recon
    img01 = np.stack([to01(a) for a in images_u8]); N = len(img01)
    gt = [m > 0 for m in masks_u8]; bbox = [roi_bbox(g) for g in gt]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)); gt_dil = [cv2.dilate(g.astype(np.uint8), k) > 0 for g in gt]
    y = meta['label'].values; pat = meta['patient_id'].values
    rows = []

    def one(scheme, source, rate, rep, seed):
        rng = np.random.default_rng(seed); imps = importance[source]
        dmg = np.empty_like(img01); ll = np.zeros(N); ps = np.zeros(N); ss = np.zeros(N)
        for i in range(N):
            lost = choose_lost(imps[i], rate, scheme, rng, cfg.burst_len)
            dmg[i], pix = apply_loss(img01[i], lost, cfg.block, recon)
            ll[i] = (pix & gt[i]).sum() / max(gt[i].sum(), 1)
            ps[i] = roi_psnr(img01[i], dmg[i], bbox[i]) if bbox[i] else np.nan
            ss[i] = roi_ssim(img01[i], dmg[i], bbox[i]) if bbox[i] else np.nan
        for rname, rcv in receivers.items():
            r = diagnose_all(rcv, dmg, chunk)
            hit = np.array([gt_dil[i][int(round(r['cy'][i])), int(round(r['cx'][i]))] for i in range(N)])
            for i in range(N):
                rows.append(dict(scheme=scheme, source=source, rate=rate, rep=rep, receiver=rname, img=i, patient_id=pat[i],
                                 y=int(y[i]), p=float(r['p'][i]), ent=float(r['ent'][i]), mi=float(r['mi'][i]),
                                 loc_conf=float(r['loc_conf'][i]), hit=bool(hit[i]), lesion_loss=float(ll[i]),
                                 psnr=float(ps[i]), ssim=float(ss[i])))

    one('clean', 'none', 0.0, 0, cfg.seed)                       # no loss at all
    for rate in rates:
        if rate == 0:
            continue
        for ci, (scheme, source) in enumerate(conditions):
            for rep in range(reps):
                one(scheme, source, rate, rep, cfg.seed + 10007 * rep + int(rate * 1000) + 97 * ci)
            if verbose:
                print(f'done: {scheme:8s} src={source:6s} rate={rate:.1f}')
    return pd.DataFrame(rows)


def summarize(df, spec=0.90):
    rows = []
    for (rcv, sch, src, rate), g in df.groupby(['receiver', 'scheme', 'source', 'rate']):
        per = np.array([[auc_safe(gr.y, gr.p), sens_at_spec(gr.y, gr.p, spec)] for _, gr in g.groupby('rep')])
        rows.append(dict(receiver=rcv, scheme=sch, source=src, rate=rate, AUC=per[:, 0].mean(), AUC_sd=per[:, 0].std(),
                         sens_at_spec=per[:, 1].mean(), loc_hit=g.hit.mean(), lesion_loss=g.lesion_loss.mean(),
                         psnr_roi=g.psnr.mean(), ssim_roi=g.ssim.mean(), mean_entropy=g.ent.mean()))
    return pd.DataFrame(rows).sort_values(['receiver', 'scheme', 'source', 'rate']).reset_index(drop=True)


def bootstrap_table(df, receiver, rate, pairs, n=1000, seed=0):
    """Patient-level bootstrap. One score per image = mean over repeats.  pairs: [((scheme, source), (scheme, source)), ...]."""
    d = df[(df.receiver == receiver) & (df.rate == rate)]
    per = d.groupby(['scheme', 'source', 'img']).agg(p=('p', 'mean'), y=('y', 'first'), pat=('patient_id', 'first')).reset_index()
    out = []
    for cond, g in per.groupby(['scheme', 'source']):
        g = g.sort_values('img'); a, lo, hi = bootstrap_ci(auc_safe, g.y.values, g.p.values, g.pat.values, n, seed)
        out.append(dict(kind='AUC', a=f'{cond[0]}/{cond[1]}', b='', value=a, ci_low=lo, ci_high=hi, p_value=np.nan))
    for ca, cb in pairs:
        ga = per[(per.scheme == ca[0]) & (per.source == ca[1])].sort_values('img')
        gb = per[(per.scheme == cb[0]) & (per.source == cb[1])].sort_values('img')
        if len(ga) == 0 or len(gb) == 0:
            continue
        v, lo, hi, p = paired_bootstrap_diff(auc_safe, ga.y.values, ga.p.values, gb.p.values, ga.pat.values, n, seed)
        out.append(dict(kind='AUC difference', a=f'{ca[0]}/{ca[1]}', b=f'{cb[0]}/{cb[1]}', value=v, ci_low=lo, ci_high=hi, p_value=p))
    return pd.DataFrame(out)


def damage_images(cfg, img01s, imps, scheme, rate, seed=0, recon=None):
    """Apply one loss realisation to every image. Returns (damaged images, list of lost-block grids)."""
    rng = np.random.default_rng(seed); dmg = np.empty_like(img01s); lost_l = []
    for i in range(len(img01s)):
        lost = choose_lost(imps[i], rate, scheme, rng, cfg.burst_len); lost_l.append(lost)
        dmg[i], _ = apply_loss(img01s[i], lost, cfg.block, recon or cfg.recon)
    return dmg, lost_l


# ============================================================ E3: adaptive loop (retransmit when uncertain, else refer)
def adaptive_loop(cfg, receiver, img01s, imps, scheme, rate, thr_ent, seed=0):
    rng = np.random.default_rng(seed); N = len(img01s)
    dmg = np.empty_like(img01s); lost_l = []
    for i in range(N):
        lost = choose_lost(imps[i], rate, scheme, rng, cfg.burst_len); lost_l.append(lost)
        dmg[i], _ = apply_loss(img01s[i], lost, cfg.block, cfg.recon)
    r1 = diagnose_all(receiver, dmg)
    p2, e2 = r1['p'].copy(), r1['ent'].copy(); extra = np.zeros(N)
    need = np.where(r1['ent'] > thr_ent)[0]
    if len(need):
        dmg2 = dmg[need].copy()
        for k, i in enumerate(need):
            req = region_blocks(r1['cy'][i], r1['cx'][i], cfg.patch / 2 + cfg.block, imps[i].shape, cfg.block)
            new_lost, n_req = retransmit(lost_l[i], req, rate, rng)
            dmg2[k], _ = apply_loss(img01s[i], new_lost, cfg.block, cfg.recon); extra[i] = n_req
        r2 = diagnose_all(receiver, dmg2)
        p2[need], e2[need] = r2['p'], r2['ent']
    n_tx = np.array([(imp >= 0).sum() for imp in imps])
    return dict(p1=r1['p'], ent1=r1['ent'], p2=p2, ent2=e2, retx=np.isin(np.arange(N), need), extra_blocks=extra,
                n_tx=n_tx, referred=e2 > thr_ent)


def adaptive_summary(y, res):
    y = np.asarray(y); keep = ~res['referred']
    acc = lambda p, m: float(((p[m] >= 0.5).astype(int) == y[m]).mean()) if m.any() else float('nan')
    return dict(AUC_first_pass=auc_safe(y, res['p1']), AUC_after_retx=auc_safe(y, res['p2']),
                retransmitted_frac=float(res['retx'].mean()), extra_bandwidth=float(res['extra_blocks'].sum() / res['n_tx'].sum()),
                referral_rate=float(res['referred'].mean()), acc_first_pass=acc(res['p1'], np.ones(len(y), bool)),
                acc_non_referred=acc(res['p2'], keep),
                AUC_non_referred=auc_safe(y[keep], res['p2'][keep]) if keep.sum() > 5 else float('nan'))


# ============================================================ E4: active learning on lesion patches
def kcenter_greedy(f_unl, f_lab, k):
    import torch
    fu, fl = torch.as_tensor(f_unl, device=DEVICE).float(), torch.as_tensor(f_lab, device=DEVICE).float()
    d = torch.cdist(fu, fl).min(1).values; sel = []
    for _ in range(k):
        j = int(torch.argmax(d)); sel.append(j)
        d = torch.minimum(d, torch.cdist(fu, fu[j:j + 1]).squeeze(1))
    return np.array(sel)


def select_samples(strategy, model, patches, lab, unl, k, rng, cfg):
    if strategy == 'random':
        return rng.choice(unl, k, replace=False)
    u = mc_predict(model, patches[unl], cfg.mc_passes)
    if strategy == 'entropy':
        return unl[np.argsort(-u['ent'])[:k]]
    if strategy == 'bald':
        return unl[np.argsort(-u['mi'])[:k]]
    if strategy == 'margin':
        return unl[np.argsort(np.abs(u['p'] - 0.5))[:k]]
    if strategy == 'coreset':
        return unl[kcenter_greedy(extract_features(model, patches[unl]), extract_features(model, patches[lab]), k)]
    raise ValueError(strategy)


def active_learning(cfg, pool_patches, pool_y, val_patches, val_y, test_patches, test_y, strategies=None, seeds=None, verbose=True):
    strategies = cfg.al_strategies if strategies is None else strategies
    seeds = cfg.al_seeds if seeds is None else seeds
    N = len(pool_y); n0 = max(int(cfg.al_init_frac * N), 10); step = max(int(cfg.al_step_frac * N), 1)
    pool_y = np.asarray(pool_y); records = []

    def train_on(lab, seed):
        dl = DataLoader(ArrDS(pool_patches[lab], pool_y[lab], True), batch_size=cfg.clf_bs, shuffle=True)
        npos = int(pool_y[lab].sum()); m, _ = fit_classifier(dl, val_patches, val_y, cfg, cfg.al_epochs, seed,
                                                           (len(lab) - npos) / max(npos, 1), verbose=False)
        return m

    def record(strat, seed, rnd, lab, m):
        p = classifier_probs(m, test_patches)
        records.append(dict(strategy=strat, seed=seed, round=rnd, n_labeled=len(lab), AUC=auc_safe(test_y, p),
                            sens_at_spec=sens_at_spec(test_y, p, 0.9), acc=float(((p >= 0.5).astype(int) == np.asarray(test_y)).mean())))

    for seed in seeds:
        rng0 = np.random.default_rng(seed)
        while True:
            init = rng0.choice(N, n0, replace=False)
            if len(np.unique(pool_y[init])) == 2:
                break
        m0 = train_on(init, seed * 1000)
        for strat in strategies:
            lab, m, rng = init.copy(), m0, np.random.default_rng(seed + 11)
            record(strat, seed, 0, lab, m)
            for r in range(1, cfg.al_rounds + 1):
                unl = np.setdiff1d(np.arange(N), lab)
                if len(unl) < step:
                    break
                lab = np.concatenate([lab, select_samples(strat, m, pool_patches, lab, unl, step, rng, cfg)])
                m = train_on(lab, seed * 1000 + r); record(strat, seed, r, lab, m)
            if verbose:
                print(f'seed {seed} | {strat:8s} | final AUC {records[-1]["AUC"]:.3f} with {len(lab)} labels')
    return pd.DataFrame(records)


def al_summary(al, target_auc=None):
    """Mean/SD AUC per label budget, area under the learning curve per strategy, labels needed for target_auc."""
    curve = al.groupby(['strategy', 'n_labeled'])['AUC'].agg(['mean', 'std']).reset_index()
    rows = []
    for s, g in curve.groupby('strategy'):
        row = dict(strategy=s, ALC=area_under_curve(g.n_labeled, g['mean']))
        if target_auc is not None:
            hit = g[g['mean'] >= target_auc]; row['labels_to_target'] = int(hit.n_labeled.iloc[0]) if len(hit) else np.nan
        rows.append(row)
    return curve, pd.DataFrame(rows)
