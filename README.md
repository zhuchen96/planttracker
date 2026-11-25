# Plant lineage tracker

Track mother–daughter families between two segmented 3D membrane microscopy frames. The tracker aligns cells across growth scales, refines the deformation from overlapping families, ranks connected daughter groups, and checks surface junctions and outlines. The result maps each assigned mother ID in the earlier mask to its daughter IDs in the later mask. Each daughter has at most one assigned mother. A missing mother key means the tracker returned no family.

This release runs the frozen MICCAI public-transfer configuration. The two pretrained checkpoints came from four adjacent transitions (1,644 annotated families) of one internal five-frame movie. Fusion thresholds were selected during development on S7 and aligned S10. Historical specimen-specific initializations and rescue runs are outside this configuration.

## Install

Use Linux, Python 3.13, and a working OpenMP runtime for LightGBM. Inference and training run on CPU. From the repository root:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(cd models && sha256sum --check SHA256SUMS)
```

## Run the public tomato example

The example fetches the original 309 MiB MorphoGraphX archive, checks its SHA-256, and converts the two surface meshes to the published working grid. For offline use, append `--archive /path/to/tomato.tar.gz` to the preparation command.

```sh
python examples/tomato/prepare_example.py work/tomato
python track.py work/tomato/frames runs/tomato tomato 0
python examples/tomato/check.py runs/tomato/predictions/tomato/T0/full.json work/tomato/families.json
```

The reference contains 488 exact families among 502 annotated mothers (97.2%). Verification also compares the digest of the complete prediction, including unannotated mothers. The images and reference annotations remain under the data providers' terms. [Kierzkowski et al., Science (2012)](https://doi.org/10.1126/science.1213100) · [MorphoGraphX data](https://morphographx.org/data/)

## Track your images

Supply one membrane-intensity TIFF and one integer instance-label TIFF per frame. They must be aligned, have identical ZYX dimensions within a frame, and use a common orientation. Prepare them on the working grid of **1.2 × 0.6 × 0.6 µm (z, y, x)** before import. Use nearest-neighbor resampling for labels. The epidermis should face low z. Use 0 for background and preserve positive cell IDs below 100,000. The six-neighbor initialization requires at least seven cells in the outer three layers of each frame. The surface tests were developed for plant tissues.

```sh
python prepare.py work/frames plantA 0 data/T0_membrane.tif data/T0_labels.tif
python prepare.py work/frames plantA 1 data/T1_membrane.tif data/T1_labels.tif
python track.py work/frames runs/plantA plantA 0
```

`prepare.py` writes copies of the two TIFFs and a derived `props.pkl` under `work/frames/plantA/T0` (or `T1`). It requires a new destination frame directory. The tracking command processes T0→T1, writes `runs/plantA/predictions/plantA/T0/full.json`, and retains intermediate branch results beside it. Rerunning the same pair checks the frame and model fingerprints before reusing a completed result. Choose a fresh run directory when changing inputs, checkpoints, or software version. Pass only trusted prepared frames and model files: the implementation uses Python pickle.

## Retrain on annotated movies

1. Prepare all input frames as above. Create one JSON file per transition containing **complete** annotated daughter sets. For example, `{"17":[3,4],"24":[9]}` means mother 17 divided into cells 3 and 4, and mother 24 persisted as cell 9. Omit unannotated mothers. Keep all frame and annotation IDs in their original label space.
2. Create a JSON manifest like the one below. Family-file paths are relative to the manifest. Use at least two distinct annotated transitions for transition-held-out cross-fitting. The annotation-assisted spline needs at least four noncoplanar center pairs, and each training table needs examples of both exact and incorrect candidate groups.
3. Choose a **new** checkpoint output directory. A source-scale run evaluates millions of connected candidate groups and is intended for a research workstation. Set `--workers` to the number of frame pairs to prepare concurrently; higher values increase memory use. Use `--keep-tables` to retain feature matrices and labels for an audit.

```json
[
  {"sequence":"plantA","earlier":0,"later":1,"families":"plantA_0_1.json"},
  {"sequence":"plantB","earlier":0,"later":1,"families":"plantB_0_1.json"}
]
```

```sh
python train.py work/frames data/transitions.json runs/new-model --workers 2
python track.py work/frames runs/new-predictions plantA 0 --models runs/new-model
```

The estimated-warp training table uses height-map alignment, raw-image block matching, and refinement from confident mask overlaps. For a new pair the search screens similarity and anisotropic height-map candidates, refines the top three and ten respectively, and selects by the number of confident families. To replay a previously selected candidate, an entry can include `"registration":{"route":"anisotropic","candidate":13}`. Indices are zero-based within a route's 60 height-map candidates. The historical source run selected anisotropic 13, anisotropic 18, similarity 39, and similarity 1 in temporal order. These indices replay the winners of the original development candidate pool; use the automatic search on new data.

The two early histogram classifiers train on estimated-warp tables. The final histogram classifier and the two LightGBM rankers use five candidate tables per transition: the estimated warp, an annotation-assisted warp, and three seeded smooth perturbations of the latter. Only an exact daughter set is positive. Unannotated mothers contribute no labels. Second-stage features use out-of-fold predictions with all variants of a transition kept together. Manifest order determines the augmentation seeds. The output includes both checkpoints and a training manifest with the checkpoint hashes. For an evaluation, withhold the intended test specimens before building this manifest.

The pretrained weights support inference without access to the private training movie. To inspect the release, see [the training command](train.py), [the inference pipeline](code/inference.py), and [the final family assignment](code/vfuse.py).
