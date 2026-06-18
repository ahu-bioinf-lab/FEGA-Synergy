# FEGA-Synergy

## Project Structure

```
FEGA-Synergy/
├── main.py                      # Main training script (5-fold cross-validation)
├── main_singel.py               # Single-fold training script
├── dataPreprocess.py            
├── DMPNN.py                    
├── mol_to_linegraph.py          
├── metrics.py                   # Evaluation metrics (MSE/RMSE/MAE/R²/Pearson/AUC/AUPR, etc.)
├── utils6异构copy.py            # Utility functions (EarlyStopping, train/validate loops, data loading)
├── co1.py                       
├── model/
│   ├── FEGA.py                  # FEGA main model definition
│   ├── ESA.py                   # Edge Set Attention module
│   ├── ESA_KIT.py               # SAB/PMA and other Set Attention building blocks
│   ├── model_utiles.py          # CellCNN, Embeddings, LayerNorm
│   ├── norm_layers.py           # Normalization layers (BatchNorm/LayerNorm)
│   ├── mlp_utils.py             # MLP utility modules
│   ├── DeepSynergy.py           # DeepSynergy baseline model
│   ├── DeepDDS.py               # DeepDDS baseline model
│   ├── build_dtn_graph.py       # Drug–target network graph construction
│   └── layers/
│       ├── PISynergy_utils.py   # Predictor, channel attention, feature noise, etc.
│       └── SynergyX_utils.py    # SynergyX helper utilities
├── dataset/
│   ├── PISynergy_dataset.py     # Custom PyG in-memory dataset
│   ├── base_InMemory_dataset.py # Base in-memory dataset class
│   ├── 药物-靶点聚合.csv         # Drug–target relation data
│   └── 通路数据映射.py           # Pathway data mapping script
├── data/
│   └── independent dataset/     # Independent test datasets
│       ├── indep2-OncologyScreen/
│       └── indep3-DrugCombDB/
├── experiment/                  # Experiment results storage
│   └── 20260318_1350_FEGA_CV_4_1/
└── 13/                          # Heterogeneous graph & auxiliary data
    ├── hetero_graph_小数据集_ALL_768_sapbert.pt
    ├── drugid小.json
    ├── proteinid全.json
    ├── drugs.pt
    ├── clines.csv
    ├── cline_de_exp.pt
    └── synergyDMPNN2.csv
```

---

## Environment Setup

### Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | **NVIDIA GeForce RTX 5080** (16 GB+ VRAM recommended) |
| CUDA | **12.8** |
| OS | Linux (Ubuntu 22.04+) / Windows 11 |

### Dependency Installation

**Python 3.10+** is recommended. Install the core deep learning packages first:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

Then install the remaining scientific computing and utility libraries:

```bash
pip install numpy==1.26.4 pandas==2.3.1 scipy==1.16.0 scikit-learn==1.7.1
pip install rdkit==2025.3.3 networkx==3.3 tqdm==4.67.1
pip install matplotlib==3.10.7 seaborn==0.13.2 adjustText==1.3.0
pip install transformers==4.53.3 tokenizers==0.21.2
pip install tensorflow==2.21.0 keras==3.14.1
pip install wandb==0.21.0 mlflow==3.11.1
pip install deepchem==2.8.0 ogb==1.3.6
pip install gensim==4.4.0 node2vec==0.5.0 pynndescent==0.6.0 umap-learn==0.12.0
pip install pytorch-lightning==2.5.2 torchmetrics==1.7.4
pip install openpyxl==3.1.5 fastapi==0.136.0 uvicorn==0.45.0
pip install bitsandbytes==0.46.1 admin-torch==0.1.0 muon-optimizer==0.1.0
pip install mygene==3.2.2 biothings_client==0.4.1 gseapy==1.1.10
pip install PubChemPy==1.0.5 sympy==1.13.3 littleutils==0.2.4
pip install gunicorn==25.3.0 Flask==3.1.3 flask-cors==6.0.2
pip install huey==2.6.0 alembic==1.18.4 SQLAlchemy==2.0.49 psutil==7.0.0
pip install sentry-sdk==2.33.0 python-dotenv==1.2.2 yacs==0.1.8
```

Or install all dependencies at once:

```bash
pip install absl-py==2.4.0 adjustText==1.3.0 admin-torch==0.1.0 aiohttp==3.12.14 \
  alembic==1.18.4 annotated-types==0.7.0 anyio==4.11.0 astunparse==1.6.3 \
  attrs==25.3.0 biothings_client==0.4.1 bitsandbytes==0.46.1 blinker==1.9.0 \
  cachetools==7.0.6 certifi==2025.7.14 cffi==2.0.0 charset-normalizer==3.4.2 \
  click==8.2.1 cloudpickle==3.1.2 contourpy==1.3.3 cryptography==46.0.7 \
  cycler==0.12.1 Cython==3.1.2 databricks-sdk==0.103.0 deepchem==2.8.0 \
  docker==7.1.0 et_xmlfile==2.0.0 fastapi==0.136.0 filelock==3.13.1 Flask==3.1.3 \
  flask-cors==6.0.2 flatbuffers==25.12.19 fonttools==4.60.1 frozenlist==1.7.0 \
  fsspec==2024.6.1 gast==0.7.0 gensim==4.4.0 GitPython==3.1.44 google-auth==2.49.2 \
  google-pasta==0.2.0 graphene==3.4.3 graphql-core==3.2.8 graphql-relay==3.2.0 \
  greenlet==3.4.0 grpcio==1.80.0 gseapy==1.1.10 gunicorn==25.3.0 h5py==3.14.0 \
  hf-xet==1.1.5 httpcore==1.0.9 httpx==0.28.1 huey==2.6.0 huggingface-hub==0.33.4 \
  idna==3.10 ijson==3.4.0.post0 importlib_metadata==8.7.1 itsdangerous==2.2.0 \
  Jinja2==3.1.4 joblib==1.5.1 keras==3.14.1 kiwisolver==1.4.9 \
  lightning-utilities==0.14.3 littleutils==0.2.4 llvmlite==0.47.0 Mako==1.3.11 \
  markdown-it-py==4.2.0 MarkupSafe==2.1.5 matplotlib==3.10.7 mdurl==0.1.2 \
  ml_dtypes==0.5.4 mlflow==3.11.1 mlflow-skinny==3.11.1 mpmath==1.3.0 \
  multidict==6.6.3 muon-optimizer==0.1.0 mygene==3.2.2 namex==0.1.0 ndex2==3.11.0 \
  networkx==3.3 node2vec==0.5.0 numba==0.65.1 numpy==1.26.4 ogb==1.3.6 \
  openpyxl==3.1.5 opentelemetry-api==1.41.0 opentelemetry-sdk==1.41.0 \
  opt_einsum==3.4.0 optree==0.19.1 outdated==0.2.2 packaging==25.0 pandas==2.3.1 \
  pillow==11.0.0 platformdirs==4.3.8 prettytable==3.17.0 propcache==0.3.2 \
  protobuf==6.31.1 psutil==7.0.0 PubChemPy==1.0.5 pyarrow==23.0.1 pyasn1==0.6.3 \
  pyasn1_modules==0.4.2 pycparser==3.0 pydantic==2.11.7 pydantic_core==2.33.2 \
  Pygments==2.20.0 pynndescent==0.6.0 pyparsing==3.2.3 python-dateutil==2.9.0.post0 \
  python-dotenv==1.2.2 pytorch-lightning==2.5.2 pytz==2025.2 PyYAML==6.0.2 \
  rdkit==2025.3.3 regex==2024.11.6 requests==2.32.4 requests-toolbelt==1.0.0 \
  rich==15.0.0 safetensors==0.5.3 scikit-learn==1.7.1 scipy==1.16.0 seaborn==0.13.2 \
  sentry-sdk==2.33.0 setuptools==78.1.1 six==1.17.0 skops==0.14.0 smart_open==7.5.0 \
  sniffio==1.3.1 SQLAlchemy==2.0.49 sqlparse==0.5.5 starlette==1.0.0 sympy==1.13.3 \
  tensorflow==2.21.0 termcolor==3.3.0 threadpoolctl==3.6.0 tokenizers==0.21.2 \
  torch==2.7.1 torch-geometric==2.6.1 torch-scatter==2.1.2 \
  torch-sparse==0.6.18 torchaudio==2.7.1 torchmetrics==1.7.4 \
  torchvision==0.22.1 tqdm==4.67.1 transformers==4.53.3 triton==3.3.1 \
  typing_extensions==4.12.2 typing-inspection==0.4.2 tzdata==2025.2 \
  umap-learn==0.12.0 urllib3==2.5.0 uvicorn==0.45.0 wandb==0.21.0 wcwidth==0.6.0 \
  Werkzeug==3.1.8 wheel==0.45.1 wrapt==2.0.1 yacs==0.1.8 yarl==1.20.1 zipp==3.23.1
```

> **Note**: `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse`, `torchaudio`, and `torchvision` must be installed with wheels matching your actual CUDA version (cu128).

---

## Quick Start

### 1. Data Preparation

Ensure the following data files are ready:

- **Heterogeneous graph**: `13/hetero_graph_小数据集_ALL_768_sapbert.pt`
- **Drug ID mapping**: `13/drugid小.json`
- **Protein ID mapping**: `13/proteinid全.json`
- **Drug features**: `13/drugs.pt`
- **Cell line features**: `13/cline_de_exp.pt`
- **Synergy data**: `13/synergyDMPNN2.csv`
- **Independent test sets**: located under `data/independent dataset/`

### 2. Training

```bash
# Train the FEGA model on the OncologyScreen dataset (5-fold cross-validation)
python main.py \
  --model FEGA \
  --dataset_name indep2-OncologyScreen \
  --celldataset 2 \
  --batch_size 128 \
  --lr 0.0001 \
  --epochs 300 \
  --patience 50 \
  --omic exp,mut,cn,eff,dep,met \
  --device cuda:0 \
  --seed 0
```

---

## Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `PISynergy` | Model selection: `PISynergy` (FEGA), `DeepSynergy`, `DeepDDSGCNNet` |
| `--dataset_name` | str | `indep2-OncologyScreen` | Dataset: `indep0-oneil`, `indep1-almanac`, `indep2-OncologyScreen`, `indep3-DrugCombDB` |
| `--celldataset` | int | `2` | Gene set selection: 1=18498g, 2=4079g, 3=963g |
| `--omic` | str | `exp,mut,cn,eff,dep,met` | Omics data types to include |
| `--batch_size` | int | `128` | Batch size |
| `--lr` | float | `0.0001` | Learning rate |
| `--epochs` | int | `300` | Maximum training epochs |
| `--patience` | int | `50` | Early stopping patience |
| `--seed` | int | `0` | Random seed |
| `--device` | str | `cuda:0` | Computation device |
| `--weight_decay` | float | `1e-3` | Weight decay |
| `--mode` | str | `train` | Running mode: `train` or `test` |



## Evaluation Metrics

The model evaluates classification performance using the following metrics:

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under the ROC Curve |
| **AUPR** | Area Under the Precision-Recall Curve |
| **Accuracy** | Classification accuracy |
| **Precision** | Precision score |
| **Recall** | Recall score |
| **F1 Score** | Harmonic mean of Precision and Recall |

For regression tasks, MSE, RMSE, MAE, R², Pearson, and Spearman metrics are also supported.

---
