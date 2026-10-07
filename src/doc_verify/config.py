from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAD_SYN_DIR = DATA_DIR / "sad_syn"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

IMAGE_SIZE = 224
SPLIT = (0.70, 0.15, 0.15)
SEED = 42
BATCH_SIZE = 32
