"""Step 5-6: packetization, semantic importance, loss models, reconstruction. Pure numpy/OpenCV (no torch)."""
import numpy as np, cv2


# ------------------------------------------------------------------ blocks
def tx_block_mask(img01, block, thr=0.03, min_frac=0.05):
    """Blocks that contain breast tissue. All-black background blocks are never transmitted (nothing to lose)."""
    H, W = img01.shape
    ny, nx = H // block, W // block
    fg = (img01[:ny * block, :nx * block] > thr).reshape(ny, block, nx, block).mean(axis=(1, 3))
    return fg > min_frac


def block_importance(prob, tx, block, thr=0.5, near=1):
    """Per-block priority from a lesion probability map: 2 = lesion, 1 = near lesion, 0 = other tissue, -1 = not sent."""
    ny, nx = tx.shape
    pm = prob[:ny * block, :nx * block].astype(np.float32).reshape(ny, block, nx, block).max(axis=(1, 3))
    t = thr if pm.max() >= thr else 0.5 * pm.max()           # if the localizer is unsure, protect its best guess anyway
    high = (pm >= max(t, 1e-6)) & tx
    nearm = cv2.dilate(high.astype(np.uint8), np.ones((2 * near + 1, 2 * near + 1), np.uint8)) > 0
    imp = np.where(high, 2, np.where(nearm, 1, 0)).astype(np.int8)
    imp[~tx] = -1
    return imp


def uniform_importance(tx):
    imp = np.zeros(tx.shape, np.int8); imp[~tx] = -1
    return imp


# ------------------------------------------------------------------ loss models
def _burst(idx, k, rng, mean_len):
    n = len(idx); chosen = np.zeros(n, bool); cnt = 0
    while cnt < k:
        start = int(rng.integers(0, n)); run = max(1, int(rng.geometric(1.0 / mean_len)))
        for j in range(start, min(n, start + run)):
            if not chosen[j]:
                chosen[j] = True; cnt += 1
            if cnt >= k:
                break
    return idx[chosen]


def choose_lost(imp, rate, scheme, rng, burst_len=6.0):
    """Which transmitted blocks are lost. Every scheme loses the SAME NUMBER of blocks (equal bandwidth).
    uniform  : independent random loss
    burst    : losses arrive in contiguous runs (raster order)
    semantic : priority-aware network, lowest-importance blocks are dropped first"""
    flat = imp.ravel(); idx = np.flatnonzero(flat >= 0); n = len(idx); k = int(round(rate * n))
    lost = np.zeros(flat.size, bool)
    if k > 0 and n > 0:
        if scheme == 'uniform':
            sel = rng.choice(idx, k, replace=False)
        elif scheme == 'burst':
            sel = _burst(idx, k, rng, burst_len)
        elif scheme == 'semantic':
            order = np.lexsort((rng.random(n), flat[idx]))     # primary key importance, random tie-break
            sel = idx[order[:k]]
        else:
            raise ValueError(scheme)
        lost[sel] = True
    return lost.reshape(imp.shape)


def blocks_to_pixels(b, block, shape):
    pix = np.zeros(shape, bool)
    up = np.repeat(np.repeat(b, block, axis=0), block, axis=1)
    pix[:up.shape[0], :up.shape[1]] = up
    return pix


# ------------------------------------------------------------------ reconstruction at the receiver
def apply_loss(img01, lost, block, method='mean'):
    """Return (reconstructed image, boolean pixel mask of lost area)."""
    pix = blocks_to_pixels(lost, block, img01.shape)
    out = img01.copy()
    if not pix.any():
        return out, pix
    if method == 'zero':
        out[pix] = 0.0
    elif method == 'mean':
        recv = (~pix) & (img01 > 0.03)
        out[pix] = float(img01[recv].mean()) if recv.any() else 0.0
    elif method == 'inpaint':
        u8 = np.clip(img01 * 255, 0, 255).astype(np.uint8)
        out = cv2.inpaint(u8, pix.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    else:
        raise ValueError(method)
    return out, pix


# ------------------------------------------------------------------ retransmission (receiver asks for lost blocks)
def region_blocks(cy, cx, half_px, tx_shape, block):
    """Boolean block grid covering a square window around (cy, cx)."""
    ny, nx = tx_shape
    r = np.zeros(tx_shape, bool)
    by0 = max(int((cy - half_px) // block), 0); by1 = min(int((cy + half_px) // block) + 1, ny)
    bx0 = max(int((cx - half_px) // block), 0); bx1 = min(int((cx + half_px) // block) + 1, nx)
    r[by0:by1, bx0:bx1] = True
    return r


def retransmit(lost, request, rate, rng):
    """One retransmission round: each requested lost block is lost again with the channel loss rate."""
    asked = lost & request
    ok = asked & (rng.random(lost.shape) >= rate)
    return lost & ~ok, int(asked.sum())
