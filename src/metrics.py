"""Evaluation metrics, patient-level bootstrap statistics, calibration and selective-prediction measures."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_safe(y, s):
    y = np.asarray(y); return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')


def sens_at_spec(y, s, spec=0.90):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y, s)
    ok = (1 - fpr) >= spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def pauc(y, s, max_fpr=0.1):
    y = np.asarray(y)
    return float(roc_auc_score(y, s, max_fpr=max_fpr)) if len(np.unique(y)) == 2 else float('nan')


def _group_index(groups):
    d = {}
    for i, g in enumerate(groups):
        d.setdefault(g, []).append(i)
    return list(d.values())


def bootstrap_ci(fn, y, s, groups, n=1000, seed=0, alpha=0.05):
    """Patient-level bootstrap CI of fn(y, s). Resamples PATIENTS, not images."""
    y, s = np.asarray(y), np.asarray(s); gi = _group_index(groups); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        pick = rng.integers(0, len(gi), len(gi)); idx = np.concatenate([gi[k] for k in pick])
        v = fn(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    return float(fn(y, s)), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(fn, y, s_a, s_b, groups, n=1000, seed=0):
    """Patient-level paired bootstrap of fn(a) - fn(b): (difference, ci_low, ci_high, two-sided p)."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    gi = _group_index(groups); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        pick = rng.integers(0, len(gi), len(gi)); idx = np.concatenate([gi[k] for k in pick])
        a, b = fn(y[idx], s_a[idx]), fn(y[idx], s_b[idx])
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    d = np.array(d)
    tie = 0.5 * (d == 0).mean()
    p = 2 * min((d < 0).mean() + tie, (d > 0).mean() + tie)
    return float(fn(y, s_a) - fn(y, s_b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(min(p, 1.0))


# ------------------------------------------------------------------ calibration and uncertainty quality
def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p))


def fit_temperature(logits, y, grid=np.linspace(0.25, 5.0, 96)):
    y = np.asarray(y, float); best, bt = 1e18, 1.0
    for t in grid:
        p = np.clip(1 / (1 + np.exp(-np.asarray(logits) / t)), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, bt = nll, t
    return float(bt)


def apply_temperature(logits, t):
    return 1 / (1 + np.exp(-np.asarray(logits) / t))


def risk_coverage(y, p, unc, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Error rate on the cases kept after referring the most uncertain ones to an expert."""
    y, p, unc = np.asarray(y), np.asarray(p), np.asarray(unc)
    order = np.argsort(unc, kind='stable'); wrong = ((p >= 0.5).astype(int) != y).astype(float)
    rows = []
    for c in coverages:
        keep = order[:max(int(round(c * len(order))), 1)]
        rows.append(dict(coverage=c, error=float(wrong[keep].mean()), accuracy=float(1 - wrong[keep].mean()),
                         auc=auc_safe(y[keep], p[keep])))
    return rows


def aurc(y, p, unc):
    y, p, unc = np.asarray(y), np.asarray(p), np.asarray(unc)
    order = np.argsort(unc, kind='stable'); wrong = ((p >= 0.5).astype(int) != y).astype(float)[order]
    return float(np.mean(np.cumsum(wrong) / np.arange(1, len(wrong) + 1)))


def error_detection_auroc(y, p, unc):
    wrong = ((np.asarray(p) >= 0.5).astype(int) != np.asarray(y)).astype(int)
    return auc_safe(wrong, unc)


# ------------------------------------------------------------------ image fidelity on the lesion region
def roi_bbox(mask, margin=8):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (max(ys.min() - margin, 0), min(ys.max() + 1 + margin, mask.shape[0]),
            max(xs.min() - margin, 0), min(xs.max() + 1 + margin, mask.shape[1]))


def roi_psnr(a, b, bbox):
    y0, y1, x0, x1 = bbox
    mse = float(np.mean((a[y0:y1, x0:x1] - b[y0:y1, x0:x1]) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def roi_ssim(a, b, bbox):
    try:
        from skimage.metrics import structural_similarity
        y0, y1, x0, x1 = bbox; ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        win = min(7, min(ca.shape) - (1 - min(ca.shape) % 2))
        return float(structural_similarity(ca, cb, data_range=1.0, win_size=max(win, 3)))
    except Exception:
        return float('nan')


# ------------------------------------------------------------------ active learning
def area_under_curve(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.trapz(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else float(ys[0])
