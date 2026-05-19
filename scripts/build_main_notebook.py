"""Generate main.ipynb from a Python template (no manual notebook editing)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# UNO Vision - main pipeline\n\n"
    "Orchestration notebook. All logic lives in `src/`; this notebook only wires modules together.\n\n"
    "See [docs/superpowers/specs/2026-05-19-main-ipynb-pipeline-design.md](docs/superpowers/specs/2026-05-19-main-ipynb-pipeline-design.md)."
))

cells.append(nbf.v4.new_markdown_cell("## A. Setup"))
cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path.cwd()))\n\n"
    "import torch\n"
    "import pandas as pd\n\n"
    "from config import (\n"
    "    TEST_IMAGES, SUBMISSIONS_DIR,\n"
    "    CNN_CHECKPOINT, YOLO_CHECKPOINT,\n"
    "    SYMBOL_CROPS_DIR, SYMBOL_CROPS_LABELS,\n"
    ")\n"
    "from src.player_zones import load_zone_fracs, center_roi_frac\n\n"
    "PREPARE_DATA = False\n"
    "TRAIN_CNN = False\n"
    "TRAIN_YOLO = False\n"
    "RUN_BACKEND_CLASSICAL = True\n"
    "RUN_BACKEND_YOLO = True\n"
    "RUN_VALIDATION = True\n\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "zone_fracs = load_zone_fracs()\n"
    "center_roi = center_roi_frac(zone_fracs)\n"
    "print('device:', device)\n"
    "print('zones:', zone_fracs)\n"
    "print('center ROI:', center_roi)"
))

cells.append(nbf.v4.new_markdown_cell("## B. Symbol crop preparation"))
cells.append(nbf.v4.new_code_cell(
    "if PREPARE_DATA:\n"
    "    !.venv/bin/python scripts/prepare_symbol_crops.py"
))

cells.append(nbf.v4.new_markdown_cell("## C. Symbol CNN training"))
cells.append(nbf.v4.new_code_cell(
    "if TRAIN_CNN:\n"
    "    !.venv/bin/python hybric_detection_utils/train_uno.py train \\\n"
    "        --image_dir data/symbol_crops \\\n"
    "        --labels data/symbol_crops/labels.csv \\\n"
    "        --output models/uno_symbol_cnn.pt \\\n"
    "        --epochs 30 --batch_size 64"
))

cells.append(nbf.v4.new_markdown_cell("## D-E. Detection backends"))
cells.append(nbf.v4.new_code_cell(
    "from src.classical_backend import detect_cards_classical\n"
    "from src.yolo_backend import detect_cards_yolo\n"
    "from src.card_labeler import load_cnn\n\n"
    "cnn, cnn_classes = load_cnn(str(CNN_CHECKPOINT), device=device)\n"
    "print('CNN classes:', cnn_classes)"
))

cells.append(nbf.v4.new_markdown_cell("## E. YOLO training (gated)"))
cells.append(nbf.v4.new_code_cell(
    "if TRAIN_YOLO:\n"
    "    !.venv/bin/python scripts/train_yolo.py"
))

cells.append(nbf.v4.new_markdown_cell("## K. Run submission for both backends"))
cells.append(nbf.v4.new_code_cell(
    "from src.orchestrator import run_submission\n\n"
    "if RUN_BACKEND_CLASSICAL:\n"
    "    run_submission(\n"
    "        image_dir=TEST_IMAGES,\n"
    "        out_csv=SUBMISSIONS_DIR / 'classical.csv',\n"
    "        detect_fn=detect_cards_classical,\n"
    "        cnn=cnn, classes=cnn_classes, device=device,\n"
    "    )\n\n"
    "if RUN_BACKEND_YOLO:\n"
    "    run_submission(\n"
    "        image_dir=TEST_IMAGES,\n"
    "        out_csv=SUBMISSIONS_DIR / 'yolo.csv',\n"
    "        detect_fn=detect_cards_yolo,\n"
    "        cnn=cnn, classes=cnn_classes, device=device,\n"
    "    )"
))

cells.append(nbf.v4.new_markdown_cell("## Backend agreement check"))
cells.append(nbf.v4.new_code_cell(
    "if RUN_BACKEND_CLASSICAL and RUN_BACKEND_YOLO:\n"
    "    a = pd.read_csv(SUBMISSIONS_DIR / 'classical.csv').set_index('image_id')\n"
    "    b = pd.read_csv(SUBMISSIONS_DIR / 'yolo.csv').set_index('image_id')\n"
    "    same = (a == b).all(axis=1)\n"
    "    print(f'Row-level agreement: {same.mean():.3f} ({same.sum()}/{len(same)})')\n"
    "    for col in a.columns:\n"
    "        agree = (a[col] == b[col]).mean()\n"
    "        print(f'  {col}: {agree:.3f}')"
))

cells.append(nbf.v4.new_markdown_cell("## L. Validation on synthetic held-out slice"))
cells.append(nbf.v4.new_code_cell(
    "from src.validation import evaluate_backend\n\n"
    "if RUN_VALIDATION:\n"
    "    if RUN_BACKEND_CLASSICAL:\n"
    "        r = evaluate_backend(detect_fn=detect_cards_classical, cnn=cnn,\n"
    "                              classes=cnn_classes, device=device)\n"
    "        print('CLASSICAL:', r)\n"
    "    if RUN_BACKEND_YOLO:\n"
    "        r = evaluate_backend(detect_fn=detect_cards_yolo, cnn=cnn,\n"
    "                              classes=cnn_classes, device=device)\n"
    "        print('YOLO:', r)"
))

nb["cells"] = cells
nbf.write(nb, "main.ipynb")
print("Wrote main.ipynb")
