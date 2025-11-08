# WWW: A Unified Framework for Explaining What, Where and Why of Neural Networks by Interpretation of Neuron Concepts

[![🦢 - Paper](https://img.shields.io/badge/🦢-Paper-red)](https://arxiv.org/abs/2402.18956)

This is the official source code for [WWW: A Unified Framework for Explaining What, Where and Why of Neural Networks by Interpretation of Neuron Concepts] 
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2024

## Usage
## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)
* [CLIP](https://github.com/openai/CLIP)
* [timm](https://github.com/huggingface/pytorch-image-models)

#### dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/ILSVRC-2012/train` and  `./datasets/ILSVRC-2012/val`, respectively.

#### Pre-trained model
For ImageNet, the model we used in the paper is the pre-trained ResNet-50 and vit is provided by Pytorch and timm. The download process
will start upon running.
For places365, please download `http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar` and place in the `./utils` folder.

## Precompute
WWW need precomputing for calculate Shapley value approximation.
Run `./extract_shap.py`.

For ImageNet with ResNet-50 experiment we placed calcualted Class-wise Shapley value in 
`./utils/RN50_ImageNet_class_shap.pkl`.

## Demo
### Example image selection
Run `./example_selection.py`.

### Concept discovery
Run `./concept_matching.py`.

### Heatmap generation
Run `./image_heatmap.py`.


### Quick demo (Tiny-ImageNet substitute)
If you don't have the full ImageNet dataset, you can run a quick end-to-end smoke demo using Tiny-ImageNet (200 classes). The commands below show a CPU-only workflow that reproduces the repository demo steps on a small dataset and saves the main artifacts.

Note: CPU runs are much slower than GPU. The commands were used during testing with limited numbers of examples (see the `--max_images` / `--num_example` flags).

1. Create and activate a virtual environment (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install CPU PyTorch and required Python packages:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
# Note: If you prefer to manually install CLIP or specific versions, you can still do:
# pip install git+https://github.com/openai/CLIP.git tqdm opencv-python matplotlib nltk ftfy regex
python -c "import nltk; nltk.download('wordnet')"
```

3. Download and extract Tiny-ImageNet (example):

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O datasets/tiny-imagenet-200.zip
unzip datasets/tiny-imagenet-200.zip -d datasets/
```

4. Run the pipeline with limited examples (adjust flags to increase/decrease work):

```bash
# compute class-level shap scores (limited)
python extract_shap.py --data_root ./datasets/tiny-imagenet-200/train --shap_save_root ./utils/heat/class_shap.pkl --max_images 200

# select example images
python example_selection.py --data_root ./datasets/tiny-imagenet-200/small_train --num_example 8 --num_act 1 --save_root ./utils --img_save_root ./images/example_val_fc

# perform concept matching (uses CLIP/text templates)
python concept_matching.py --img_save_root ./images/example_val_fc --num_example 8 --data 1

# generate heatmaps from selected examples
python image_heatmap.py --example_root ./images/example_val_fc --num_example 8 --heatmap_save_root ./heatmap --util_root ./utils --map_root ./heatmap_info
```

5. Main outputs (example locations):

- `./utils/heat/class_shap.pkl` — class-wise shap/taylor scores (small-run, e.g. 200 x 2048)
- `./images/example_val_fc/` — saved example images produced by `example_selection.py`
- `./utils/img_features_8.pkl`, `./utils/words_only.pkl`, `./utils/concept_adaptive_sim_8_*.pkl` — concept matching artifacts
- `./heatmap/` — per-class heatmap images (folders per class)
- `./heatmap_info/c_heatmap.pkl`, `./heatmap_info/s_heatmap.pkl`, `./heatmap_info/cos.pkl` — heatmap arrays and cosine similarity results


### Using the `run_demo.sh` wrapper

A convenience script `run_demo.sh` is included to automate the Tiny-ImageNet smoke demo end-to-end. It installs requirements (if a virtualenv is active), runs the four pipeline steps and puts outputs under `./utils`, `./images`, and `./heatmap`.

Usage examples (run from repository root):

```bash
# basic (assumes you created and activated a venv and installed CPU torch first)
./run_demo.sh

# set environment variables to tune the demo
MAX_IMAGES=200 NUM_EXAMPLE=8 ./run_demo.sh

# download Tiny-ImageNet and create a small subset before running
# DOWNLOAD_TINY=1 CREATE_SMALL=1 MAX_IMAGES=200 NUM_EXAMPLE=8 ./run_demo.sh
```

Notes:
- If you don't have `.venv`, the script will attempt to install `requirements.txt` into your current environment.
- To install all demo dependencies (including CPU PyTorch), follow the README section above and/or run:

```bash
# install CPU PyTorch first (required for CPU-only runs)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# then install the remaining packages listed in requirements.txt
pip install -r requirements.txt
```

### Artifacts produced (where to find them)

After running the demo (or the individual scripts), you can find the main artifacts here:

- Pickles and CLIP/concept outputs: `./utils/`
	- Example files present in this workspace: `img_features_8.pkl`, `words_only.pkl`, `concept_adaptive_sim_8_*.pkl`, `RN50_ImageNet_class_shap.pkl` (if available)
- Class shap/taylor scores: `./utils/heat/class_shap.pkl` (small-run example ~3 MB)
- Selected example images: `./images/example_val_fc/` (one folder per class, many JPGs)
- Per-class heatmap images: `./heatmap/` (subfolders for each class id, images inside)
- Heatmap arrays / similarity info: `./heatmap_info/` (e.g. `c_heatmap.pkl`, `s_heatmap.pkl`, `cos.pkl` when produced)

###TODO
generate a simple manifest with the included tool:

```bash
# produce a quick manifest (no checksums)
python tools/produce_artifact_manifest.py --out demo_artifacts.json

# produce a manifest with sha256 checksums (slower)
python tools/produce_artifact_manifest.py --out demo_artifacts_with_checksums.json --checksums
```

### Developer / tests

If you want to run the small unit test that verifies the manifest tool and perform quick developer checks, install the requirements (this includes `pytest`):

```bash
# install demo dependencies and test tools
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt

# run the manifest tool (quick)
python tools/produce_artifact_manifest.py --out demo_artifacts.json

# run the pytest suite for the manifest tool (disable plugin autoload to avoid unrelated system plugins)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_manifest.py
```

This will produce `demo_artifacts.json` and run the small test under `tests/` which ensures the manifest tool is working in a clean temporary folder.

### Dataset references and notes

- Tiny-ImageNet (closest drop-in substitute by layout/appearance):
	- Link: http://cs231n.stanford.edu/tiny-imagenet-200.zip
	- Size: ~260 MB compressed, 200 classes, 500 train images per class (64×64 images)
	- Notes: images are smaller than ImageNet (64×64). Models in this repo expect 224×224 input but the scripts use transforms (Resize/CenterCrop) that upsample Tiny-ImageNet to the model input size; this usually works for smoke runs but be aware that upsampling may reduce fidelity.

- Broden 1.1 (optional - used for concept labeling / network dissection):
	- Link: https://zenodo.org/record/2538594
	- Notes: use Broden only if you need per-pixel concept ground-truth labels for quantitative concept evaluation.

- Places365 (weights only used by this repo for scene-class models):
	- Weights link: http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
	- Place the downloaded `resnet18_places365.pth.tar` in the `./utils` folder as the scripts expect.

These alternatives let you run the demo and concept-discovery pipeline without the full ImageNet-1k dataset. If you plan full experiments or quantitative comparisons, use the corresponding full datasets and follow the dataset preparation steps in the original papers.



