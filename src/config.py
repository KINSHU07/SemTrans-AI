"""All experiment settings in one place. Edit paths in the notebook, not here."""
from dataclasses import dataclass, field, asdict
import json, os


@dataclass
class Cfg:
    # ------------------------------------------------------------------ paths (EDIT)
    cbis_root: str = '/content/drive/MyDrive/CBIS-DDSM'            # folder holding the 4 CSVs + downloaded DICOM folders
    work_dir: str = '/content/drive/MyDrive/mammo_semcom_work'     # caches, checkpoints, results, figures

    # ------------------------------------------------------------------ data
    max_images: int = 0            # 0 = every image found; set e.g. 200 for a quick dry run
    img_h: int = 768               # working resolution (after breast crop). Raise to 1024/1536 if RAM allows.
    img_w: int = 512
    patch: int = 192               # lesion patch side at working resolution
    patch_out: int = 224           # classifier input size
    n_workers: int = 4             # threads for DICOM reading
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ------------------------------------------------------------------ channel / transmission
    block: int = 32                # packet = block x block pixels (768x512 -> 24x16 = 384 blocks)
    loss_rates: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    reps: int = 3                  # repeats of the random loss pattern
    burst_len: float = 6.0         # mean burst length (blocks) for the bursty channel
    recon: str = 'mean'            # 'mean' | 'zero' | 'inpaint'
    near_blocks: int = 1           # blocks around the lesion counted as "near"

    # ------------------------------------------------------------------ training
    pretrained: bool = True        # ImageNet weights (needs internet on first use)
    loc_epochs: int = 20
    loc_bs: int = 4
    loc_lr: float = 3e-4
    clf_epochs: int = 15
    clf_bs: int = 32
    clf_lr: float = 1e-4
    ensemble_size: int = 3
    mc_passes: int = 20
    loss_aug_max: float = 0.5      # max packet-loss rate used for robust-training augmentation
    num_workers: int = 2

    # ------------------------------------------------------------------ active learning
    al_init_frac: float = 0.05
    al_step_frac: float = 0.03
    al_rounds: int = 6
    al_seeds: tuple = (0, 1, 2)
    al_strategies: tuple = ('random', 'entropy', 'bald', 'coreset')
    al_epochs: int = 6

    # ------------------------------------------------------------------ helpers
    def path(self, *p):
        return os.path.join(self.work_dir, *p)

    def ensure(self):
        for d in ['cache', 'models', 'results', 'figures']:
            os.makedirs(self.path(d), exist_ok=True)
        return self

    def save(self, name='config.json'):
        self.ensure()
        with open(self.path(name), 'w') as f:
            json.dump(asdict(self), f, indent=2)
