# AI Safety – Concepts and Techniques · Homework 1 · Group 4

University of Oldenburg · Winter semester 2026/27

This repository contains the code, notebooks and results for **Part 1** of Homework 1: running the autonomous-driving model **OpenDriveVLA-0.5B** on nuScenes and studying its robustness to Gaussian image noise.
The comparison with Alpamayo-R1 (Task B) and **Part 2** (STPA analysis of an in-vehicle generative-AI voice assistant) are in the report.

---

## Team

| Member | Name | Responsibilities |
|---|---|---|
| 1 | Josias Salou | Environment setup, Task A, shared runner script, repository and notebook; STPA losses and hazards |
| 2 | Frank | Task C (Gaussian noise, deviation metric) and Task E (ODD ontology, pairwise coverage); unsafe control actions |
| 3 | Corel | Task D (30 seeded samples, clean and noisy runs, robustness analysis); causal scenarios and safety requirements |
| 4 | Thomas | Task B (comparison with Alpamayo-R1), report assembly, GenAI statement, submission; AI requirements and verification plan |

## Repository structure

```
code/
  run_opendrivevla.py      Runner: runs OpenDriveVLA on any nuScenes sample, with an optional image transform (member 1)
  constraints.txt          pinned library versions
  noise_patch.py           Gaussian-noise patch on the image loader, controlled by environment variables (member 2)
  run_noise.py             runner with the noise patch enabled, used for Task C (member 2)
notebooks/                 notebooks with their saved outputs, and HTML exports
  HW1_TaskA_OpenDriveVLA_setup_member1.ipynb / .html   Task A: setup and run on samples 320 and 319
  HW1_TaskD_inference_member3.ipynb / .html            Task D: sample draw, characteristics, inference
  HW1_TaskD_analysis.ipynb / .html                     Task D: metrics, figures, conclusion
  HW1_Tache_D_membre3.ipynb                            working copy of the Task D inference notebook
docs/
  INSTALLATION_COLAB.md    step-by-step installation from scratch on Google Colab
```

Not included, because of their size or licence: the Python environment, the model weights (gated on Hugging Face) and the nuScenes data (nuScenes licence); `docs/INSTALLATION_COLAB.md` explains how to obtain them. The raw result files (JSON, CSV, images) were kept on the team's shared drive: all results are visible in the saved notebook outputs and in the report.

## Quick start

1. Open a Google Colab notebook with a **T4 GPU** (Runtime → Change runtime type).
2. Follow `docs/INSTALLATION_COLAB.md`: build the Python 3.10 environment, compile mmcv and mmdet3d, download the model and the required nuScenes images.
3. Run Task A:

```bash
cd /content/OpenDriveVLA
PYTHONPATH=/content/OpenDriveVLA:/content/OpenDriveVLA/drivevla \
  /content/env/bin/python -u /path/to/code/run_opendrivevla.py --targets 320 319 --out A_samples_320_319.json
```

4. Use the runner from another script (this is how the noise experiments plug in):

```python
from run_opendrivevla import Runner

runner = Runner()                                   # loads the model and the data once (about 1 min)
clean = runner.run_sample(320)                      # unmodified images
noisy = runner.run_sample(320, transform=my_noise)  # my_noise(img, frame, cam) -> img
print(clean['trajectory'], noisy['trajectory'])
```

`transform` receives every camera image right after loading and before the model's normalisation: a NumPy `float32` array, BGR, values 0–255, shape (900, 1600, 3), plus the sample index and the camera index (0 = front). By default only the target sample is transformed; `transform_previous_frames=True` also transforms the earlier frames of the scene.

5. Noise runs with member 2's patch: `run_noise.py` is the same runner with `noise_patch` enabled. The noise is set by environment variables: `NOISE_SIGMA` (standard deviation on the 0–255 scale), `NOISE_SEED` (default 4) and `NOISE_SAVE` (optional path of the front-camera image to save). The noise is added to all six cameras after loading, and the result is clipped to [0, 255].

## Environment

| Component | Version |
|---|---|
| Hardware | Google Colab, Tesla T4 (16 GB VRAM), about 12.7 GB RAM |
| Python (separate environment) | 3.10.22 |
| CUDA toolkit / compiler | 12.1.1 / gcc 11.4 |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2 |
| mmcv-full | 1.7.2 (compiled, `TORCH_CUDA_ARCH_LIST=7.5`) |
| mmdet / mmsegmentation / mmengine | 2.26.0 / 0.29.1 / 0.9.0 |
| mmdet3d | 1.0.0rc6 (compiled) |
| transformers | commit `1c39974a` (version used by the authors) |
| deepspeed / accelerate / tokenizers | 0.14.2 / 0.29.3 / 0.15.2 |

Inference settings: float16, PyTorch `sdpa` attention, greedy decoding (`do_sample=False`, `temperature=0`, `num_beams=1`, `max_new_tokens=512`), as in the original inference script.

### Deviations from the original OpenDriveVLA repository

- **Pinned versions** (see `code/constraints.txt`): `numpy==1.26.1`, `setuptools==69.5.1`, `yapf==0.40.1`, `opencv-python==4.9.0.80`, `matplotlib==3.8.4`, `shapely==1.8.5.post1`. The recent versions installed by pip are incompatible with PyTorch 2.1.2, mmcv 1.7 or the UniAD code. `MPLBACKEND=Agg` is set to avoid a Colab plotting backend.
- **One modified line**: `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`, line 34, `seaborn-whitegrid` → `seaborn-v0_8-whitegrid` (style renamed in recent matplotlib).
- **`flash-attn` not installed**: the inference code falls back to PyTorch `sdpa` attention.
- **Reduced nuScenes metadata**: with the full metadata, the run exceeded Colab's RAM and was killed. We kept only the keyframe entries and the annotations of the validation scenes. The model only reads these entries, so its inputs are unchanged.

| Table | Original | Reduced |
|---|---|---|
| sample_data | 2,631,083 | 409,788 |
| ego_pose | 2,631,083 | 409,788 |
| sample_annotation | 1,166,187 | 192,041 |

## Data and sample selection

- **Dataset**: nuScenes v1.0-trainval, validation split (6,019 samples), using the UniAD info file `nuscenes_infos_temporal_val.pkl` and the authors' cached planning information.
- **Indexing convention**: samples are numbered as in the OpenDriveVLA code, i.e. the validation set **sorted by timestamp**, **0-based**. Since "the 80β-th sample" can be read as 0-based or 1-based, we ran both **index 320** (main result) and **index 319**.
- **Task A**: group 4 → 80 × 4 = 320. Sample token `f38ef51177c84c2a9fa32584b79e601e`, scene `scene-0276` ("Intersection, peds, cars, scooter, parked bicycle"), 5th frame of the scene.
- **Tasks D and E**: `random.Random(4).sample(range(6019), 30)` (seed = group ID), giving
  `162, 210, 482, 544, 738, 845, 869, 1269, 1350, 1414, 1584, 1756, 1818, 1933, 2132, 2144, 2226, 2266, 2370, 2484, 2538, 2951, 3244, 3289, 3922, 4262, 4396, 4500, 5248, 5908`.
- **Temporal memory**: the model keeps a memory of earlier frames of the same scene (tracked objects, bird's-eye-view features). Before each target sample, the runner therefore replays the previous frames of its scene in order (0 to 40 frames, 20.7 on average for the 30 samples). In total, 3,696 camera images in 29 scenes are needed (about 515 MB).

## Results

### Task A

Predicted trajectories: 6 waypoints (x, y) in metres, one every 0.5 s over 3 s; y is the driving direction.

| Sample | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | 3.0 s |
|---|---|---|---|---|---|---|
| **320** (4 previous frames) | (−0.03, 0.84) | (−0.10, 2.16) | (−0.25, 3.79) | (−0.45, 5.68) | (−0.69, 7.74) | (−0.94, 9.94) |
| 319 (3 previous frames) | (0.01, 0.35) | (0.02, 1.04) | (0.06, 2.17) | (0.13, 3.68) | (0.22, 5.49) | (0.33, 7.49) |

For sample 320, the vehicle moves about 10 m forward in 3 s and accelerates gently, with a slight lateral drift of about 0.9 m, which is plausible when pulling away at an intersection with pedestrians. The run of sample 320 was repeated on another day and another Colab account, and in Task D: the trajectory was identical each time.

### Task C – noise on the assigned sample

See the report (member 2).

### Task D – 30 samples

Clean runs and Gaussian noise σ ∈ {10, 25, 50} (0–255 scale, clipped, noise seed 4, noise also applied to the previous frames of the scene). Deviation metrics between the noisy and the clean trajectory: ADE, FDE (at 3 s) and maximum lateral deviation; a deviation is counted as notable above 0.5 m laterally or 1 m at 3 s.

Because of limited GPU time, **19 of the 30 samples** (75 of the 120 planned runs) were processed before the deadline:

| σ | Samples | Mean ADE (m) [95 % CI] | Mean FDE (m) | Max FDE (m) | Max lateral (m) | Identical to clean | Notable |
|---|---|---|---|---|---|---|---|
| 10 | 19 | 0.27 [0.10, 0.48] | 0.64 | 3.48 | 0.74 | 6 / 19 | 5 / 19 |
| 25 | 19 | 0.15 [0.06, 0.25] | 0.39 | 1.77 | 0.83 | 6 / 19 | 4 / 19 |
| 50 | 18 | 0.27 [0.12, 0.47] | 0.64 | 3.48 | 1.03 | 6 / 18 | 5 / 18 |

Deviations are mostly longitudinal (planned speed), do not grow steadily with σ, and the output often jumps between a few fixed answers. Compared with the real displacement, noise makes the plans different but not clearly worse: its effect is several times smaller than the model's own error on clean images. Details, figures and limitations: `notebooks/HW1_TaskD_analysis.ipynb`. The Task D runs were made by a script `run_task_D.py`, written and launched from `notebooks/HW1_TaskD_inference_member3.ipynb`.

### Task E – ODD and pairwise coverage

See the report (member 2).

## Reproducibility

- Group ID 4; Task A index 320 (and 319); sampling and noise seed 4; 0-based indexing of the validation set sorted by timestamp.
- Deterministic decoding and pinned library versions (see above). The clean trajectory of sample 320 was reproduced identically across sessions, accounts and tasks.
- `notebooks/HW1_TaskD_analysis.ipynb` needs no GPU: it only reads the raw result files `D_resultats_bruts.json` and `D_caracteristiques_30.csv` produced by the inference notebook.

## Limitations

- Task A is a single sample; general conclusions come from Task D, which covers 19 of the 30 samples, hence wide confidence intervals.
- One noise seed, synthetic i.i.d. noise, open-loop evaluation, few night, rain or turning cases.
- The environment is specific to Google Colab; another machine requires recompiling mmcv and mmdet3d with the same fixes.

## Use of generative AI

Code written with AI assistance is marked in the header of the files concerned. `code/run_opendrivevla.py`, the data-preparation cells and `docs/INSTALLATION_COLAB.md` were written with the help of Claude (Anthropic) and checked by execution on Colab (versions displayed, number of extracted images, identical sample selection on two machines, model outputs). The full GenAI statement (tools, sections, cells, checks) is in the report.

## Acknowledgements and licences

- OpenDriveVLA: https://github.com/DriveVLA/OpenDriveVLA · model `OpenDriveVLA/OpenDriveVLA-0.5B` on Hugging Face.
- nuScenes (Motional): https://www.nuscenes.org, used under the nuScenes terms of use; the data are not redistributed here.
- UniAD (OpenDriveLab), whose info files and perception module are used by OpenDriveVLA.
