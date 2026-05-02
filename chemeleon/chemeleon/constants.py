import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PATH_CLIP_GENERAL_TEXT = os.path.join(CHECKPOINT_DIR, "clip-upy53q4b.ckpt")
PATH_CHEMELEON_GENERAL_TEXT = os.path.join(CHECKPOINT_DIR, "chemeleon-7fsg68c3.ckpt")
PATH_CLIP_COMPOSITION = os.path.join(CHECKPOINT_DIR, "clip-hlfus38h.ckpt")
PATH_CHEMELEON_COMPOSITION = os.path.join(CHECKPOINT_DIR, "chemeleon-fksq6cgp.ckpt")

CHECKPOINT_URLS = {
    "clip_general_text": "https://figshare.com/ndownloader/files/49891233",
    "chemeleon_general_text": "https://figshare.com/ndownloader/files/49891230",
    "clip_composition": "https://figshare.com/ndownloader/files/49891287",
    "chemeleon_composition": "https://figshare.com/ndownloader/files/49891284",
}

__all__ = [
    "PATH_CLIP_GENERAL_TEXT",
    "PATH_CHEMELEON_GENERAL_TEXT",
    "PATH_CLIP_COMPOSITION",
    "PATH_CHEMELEON_COMPOSITION",
    "CHECKPOINT_URLS",
]
