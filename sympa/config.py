
import torch
from pathlib import Path

PREP_PATH = Path("data/prep")
CKPT_PATH = Path("ckpt")
TENSORBOARD_PATH = Path("tensorboard")
PREPROCESSED_FILE = "preprocessed-data.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device=DEVICE)

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

INIT_EPS = 1e-3
