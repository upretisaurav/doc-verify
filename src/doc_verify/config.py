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

# test-set results are only allowed once every modelling choice is frozen; flip this deliberately, once
TEST_SET_UNLOCKED = False

# every dataset manifest (SAD-Syn and Tobacco3482) has exactly these columns; a column that does not apply is empty
MANIFEST_COLUMNS = [
    "source", "doc_type", "split", "label", "inconsistency_type", "altered_field", "altered_value_ocr_ok",
    "template", "pair_id", "seed", "strength", "copies_read", "copies_total", "mean_confidence",
    "image_path", "text_path",
]
