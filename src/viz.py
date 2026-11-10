"""Figures. Colour-blind-safe palette (Okabe-Ito)."""
import numpy as np, matplotlib
import matplotlib.pyplot as plt
import cv2

PAL = {'uniform': '#D55E00', 'burst': '#E69F00', 'semantic/pred': '#009E73', 'semantic/oracle': '#0072B2',
       'clean': '#555555', 'random': '#D55E00', 'entropy': '#009E73', 'bald': '#0072B2', 'coreset': '#CC79A7', 'margin': '#56B4E9'}


def _c(k):
    return PAL.get(k, '#444444')


def show_samples(images, masks, images_df, n=8, path=None, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(images), min(n, len(images)), replace=False)
    fig, ax = plt.subplots(2, (len(idx) + 1) // 2, figsize=(3 * ((len(idx) + 1) // 2), 8))
    for a, i in zip(ax.ravel(), idx):
        rgb = np.stack([images[i]] * 3, -1).copy()
        cnt, _ = cv2.findContours((masks[i] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnt, -1, (255, 60, 60), 2)
        a.imshow(rgb); a.axis('off'); r = images_df.iloc[i]
        a.set_title(f"{r.patient_id} | {'malignant' if r.label else 'benign'}", fontsize=9)
    for a in ax.ravel()[len(idx):]:
        a.axis('off')
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()


def plot_transmission(summary, receiver, path=None, title=''):
    s = summary[summary.receiver == receiver]; clean = s[s.scheme == 'clean']
    panels = [('AUC', 'AUC', 'AUC_sd'), ('sens_at_spec', 'Sensitivity at 90% specificity', None),
              ('loc_hit', 'Lesion localization hit rate', None), ('lesion_loss', 'Share of lesion area lost', None)]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for a, (col, ttl, sd) in zip(ax, panels):
        for (sch, src), g in s[s.scheme != 'clean'].groupby(['scheme', 'source']):
            key = sch if src == 'none' else f'{sch}/{src}'
            g = g.sort_values('rate'); xs = np.r_[0.0, g.rate.values] * 100
            ys = np.r_[clean[col].values[:1], g[col].values] if len(clean) else g[col].values
            if len(clean) == 0: xs = g.rate.values * 100
            a.plot(xs, ys, marker='o', color=_c(key), label=key)
            if sd and len(clean) and sd in g:
                e = np.r_[0, g[sd].values]; a.fill_between(xs, ys - e, ys + e, color=_c(key), alpha=0.15)
        a.set_title(ttl); a.set_xlabel('packet loss (%)'); a.grid(alpha=0.3)
    ax[0].legend(); fig.suptitle(title or f'Receiver: {receiver}')
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()


def plot_risk_coverage(curves, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in curves.items():
        cov = [r['coverage'] * 100 for r in rows]
        ax[0].plot(cov, [r['error'] for r in rows], marker='o', label=name)
        ax[1].plot(cov, [r['auc'] for r in rows], marker='o', label=name)
    ax[0].set_ylabel('error on kept cases'); ax[1].set_ylabel('AUC on kept cases')
    for a in ax:
        a.set_xlabel('coverage (% of cases kept, most uncertain referred)'); a.invert_xaxis(); a.grid(alpha=0.3)
    ax[0].legend(); ax[0].set_title('Risk-coverage'); ax[1].set_title('AUC vs coverage')
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()


def plot_reliability(y, probs, path=None, bins=10):
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot([0, 1], [0, 1], 'k--', lw=1)
    y = np.asarray(y, float)
    for name, p in probs.items():
        p = np.asarray(p); edges = np.linspace(0, 1, bins + 1); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() >= 3: xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker='o', label=name)
    ax.set_xlabel('predicted malignant probability'); ax.set_ylabel('observed malignant fraction'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()


def plot_al(al, path=None, target=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for s, g in al.groupby('strategy'):
        m = g.groupby('n_labeled')['AUC'].agg(['mean', 'std']).fillna(0)
        ax[0].plot(m.index, m['mean'], marker='o', color=_c(s), label=s)
        ax[0].fill_between(m.index, m['mean'] - m['std'], m['mean'] + m['std'], color=_c(s), alpha=0.12)
    if target: ax[0].axhline(target, ls='--', color='gray', label='target')
    ax[0].set_xlabel('labeled lesions'); ax[0].set_ylabel('test AUC'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title('Performance vs labeling effort')
    base = al[al.strategy == 'random'].set_index(['seed', 'n_labeled'])['AUC']
    for s, g in al[al.strategy != 'random'].groupby('strategy'):
        d = (g.set_index(['seed', 'n_labeled'])['AUC'] - base).groupby('n_labeled').agg(['mean', 'std']).fillna(0)
        ax[1].errorbar(d.index, d['mean'], yerr=d['std'], marker='o', capsize=3, color=_c(s), label=s)
    ax[1].axhline(0, color='k', lw=0.8); ax[1].set_xlabel('labeled lesions'); ax[1].set_ylabel('AUC gain over random')
    ax[1].set_title('Gain over random selection (mean +/- sd over seeds)'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()


def plot_loss_example(img01, heat, imp, lost_u, lost_s, block, recon_u, recon_s, path=None):
    fig, ax = plt.subplots(1, 5, figsize=(20, 5.4))
    ax[0].imshow(img01, cmap='gray', vmin=0, vmax=1); ax[0].set_title('original')
    ax[1].imshow(img01, cmap='gray', vmin=0, vmax=1); ax[1].imshow(heat, cmap='magma', alpha=0.5); ax[1].set_title('sender lesion map')
    ax[2].imshow(np.kron(imp, np.ones((block, block))), cmap='viridis'); ax[2].set_title('block priority\n(2 lesion, 1 near, 0 tissue, -1 not sent)')
    ax[3].imshow(recon_u, cmap='gray', vmin=0, vmax=1); ax[3].set_title('uniform loss')
    ax[4].imshow(recon_s, cmap='gray', vmin=0, vmax=1); ax[4].set_title('semantic loss')
    for a in ax: a.axis('off')
    plt.tight_layout()
    if path: plt.savefig(path, dpi=150)
    plt.show()
