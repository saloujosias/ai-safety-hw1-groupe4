# Installation complète sur Google Colab (GPU T4)

Ces cellules recréent l'environnement depuis zéro. Elles sont à exécuter dans l'ordre, dans un notebook Colab réglé sur **Runtime → Change runtime type → T4 GPU**.

Durée : environ 1 h 30, dont environ 40 min de compilation de mmcv. Une fois l'installation terminée, l'environnement peut être archivé (dernière cellule) puis restauré en quelques minutes.

> Usage de l'IA générative : ces cellules ont été préparées avec l'aide de Claude (Anthropic) et vérifiées par exécution sur Colab.

## 1. Code et environnement Python 3.10 séparé

Colab utilise Python 3.13 et CUDA 13, alors qu'OpenDriveVLA exige Python 3.10 et PyTorch 2.1.2.

```python
%cd /content
!git clone -q https://github.com/DriveVLA/OpenDriveVLA.git
!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content bin/micromamba
!/content/bin/micromamba create -y -q -p /content/env -c nvidia/label/cuda-12.1.1 -c conda-forge \
    python=3.10 cuda-toolkit gcc_linux-64=11 gxx_linux-64=11
!/content/env/bin/pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
!cp /content/ai-safety-hw1-groupe4/code/constraints.txt /content/constraints.txt   # ou créer le fichier à la main
!/content/env/bin/pip install -c /content/constraints.txt numpy==1.26.1 setuptools==69.5.1 yapf==0.40.1 \
    opencv-python==4.9.0.80 ninja psutil addict packaging Pillow pyyaml
```

## 2. Réglages de compilation

```python
import os
os.environ['CUDA_HOME'] = '/content/env'
os.environ['PATH'] = '/content/env/bin:' + os.environ['PATH']
os.environ['CC']  = '/content/env/bin/x86_64-conda-linux-gnu-gcc'
os.environ['CXX'] = '/content/env/bin/x86_64-conda-linux-gnu-g++'
os.environ['LIBRARY_PATH'] = '/content/env/lib'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'     # GPU T4
os.environ['MMCV_WITH_OPS'] = '1'
os.environ['MAX_JOBS'] = '2'
os.environ['MPLBACKEND'] = 'Agg'
```

## 3. mmcv (compilé), mmdet, mmdet3d (compilé)

```python
%cd /content/OpenDriveVLA/third_party/mmcv_1_7_2
!/content/env/bin/pip install -c /content/constraints.txt . --no-build-isolation 2>&1 | tail -3
!/content/env/bin/pip install -c /content/constraints.txt mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 \
    motmetrics==1.4.0 casadi==3.6.0 scipy==1.10.1 scikit-image==0.19.3 fsspec
%cd /content/OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6
!/content/env/bin/pip install -c /content/constraints.txt . --no-build-isolation 2>&1 | tail -3
```

## 4. Dépendances du modèle et corrections

```python
%cd /content/OpenDriveVLA
!/content/env/bin/pip install -c /content/constraints.txt --no-build-isolation \
    "transformers @ git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4" \
    "tokenizers~=0.15.2" accelerate==0.29.3 deepspeed==0.14.2 peft==0.4.0 "sentencepiece~=0.1.99" \
    einops==0.6.1 einops-exts==0.0.4 timm shortuuid ftfy pydantic==1.10.8 scikit-learn==1.2.2 \
    pytorch-lightning==1.2.5 torchmetrics==0.11.4 pyquaternion shapely descartes prettytable \
    matplotlib==3.8.4 huggingface_hub
!/content/env/bin/pip install "shapely==1.8.5.post1"
# Seule modification du dépôt : nom d'un style matplotlib renommé dans les versions récentes
!sed -i "s/plt.style.use('seaborn-whitegrid')/plt.style.use('seaborn-v0_8-whitegrid')/" \
    /content/OpenDriveVLA/projects/mmdet3d_plugin/datasets/eval_utils/map_api.py
```

Vérification :

```python
!PYTHONPATH=/content/OpenDriveVLA:/content/OpenDriveVLA/drivevla /content/env/bin/python -c \
 "import torch, mmcv; from llava.model.builder import load_pretrained_model; import projects.mmdet3d_plugin; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, 'OK')"
```

## 5. Modèle (accès contrôlé sur Hugging Face)

Demander l'accès sur https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B, puis créer un jeton en lecture.

```python
from getpass import getpass
os.environ['HF_TOKEN'] = getpass("Token Hugging Face : ")
!/content/env/bin/huggingface-cli download OpenDriveVLA/OpenDriveVLA-0.5B \
    --local-dir /content/OpenDriveVLA/checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct
```

## 6. Données nuScenes (seulement les images nécessaires)

Les données nuScenes sont soumises à la licence nuScenes et ne sont pas redistribuées dans ce dépôt. Un compte sur nuscenes.org est nécessaire pour accepter la licence. La cellule ne télécharge que les images de l'échantillon 320 (et 319), des 30 échantillons de la tâche D et des instants précédents de leurs scènes : 3 696 images, environ 515 Mo.

```python
import os, pickle, random, collections, tarfile, urllib.request, time
L = '/content/data'
os.makedirs(f'{L}/infos', exist_ok=True); os.makedirs(f'{L}/nuscenes', exist_ok=True)
CDN = 'https://d36yt3mvayqw5m.cloudfront.net/public/v1.0'
!wget -q -O {L}/infos/nuscenes_infos_temporal_val.pkl https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl
!pip install -q gdown && cd {L}/nuscenes && gdown -q 16X0_-v-iXP9hVLNaDMmIiGhZKj24YOnb
!wget -q -O /content/meta.tgz {CDN}/v1.0-trainval_meta.tgz && tar -xzf /content/meta.tgz -C {L}/nuscenes
!wget -q -O /content/maps.zip {CDN}/nuScenes-map-expansion-v1.3.zip && unzip -q -o /content/maps.zip -d {L}/nuscenes/maps

infos = sorted(pickle.load(open(f'{L}/infos/nuscenes_infos_temporal_val.pkl', 'rb'))['infos'], key=lambda e: e['timestamp'])
targets = [320, 319] + random.Random(4).sample(range(len(infos)), 30)
by_scene = collections.defaultdict(list)
for i, x in enumerate(infos): by_scene[x['scene_token']].append(i)
needed = {i for t in targets for i in by_scene[infos[t]['scene_token']] if infos[i]['frame_idx'] <= infos[t]['frame_idx']}
remaining = {c['data_path'][c['data_path'].index('samples/'):] for i in needed for c in infos[i]['cams'].values()}
del infos
for part in range(1, 11):
    if not remaining: break
    with urllib.request.urlopen(f'{CDN}/v1.0-trainval{part:02d}_keyframes.tgz') as r, tarfile.open(fileobj=r, mode='r|gz') as tar:
        for m in tar:
            name = m.name.lstrip('./')
            if m.isfile() and name in remaining:
                os.makedirs(os.path.dirname(f'{L}/nuscenes/{name}'), exist_ok=True)
                open(f'{L}/nuscenes/{name}', 'wb').write(tar.extractfile(m).read())
                remaining.discard(name)
    print(f'Partie {part} -> il reste {len(remaining)} images')
!ln -sfn /content/data /content/OpenDriveVLA/data
```

## 7. Allègement des métadonnées (RAM de Colab gratuit)

Avec les métadonnées complètes, le chargement dépasse les 12,7 Go de RAM. Le modèle ne lit que les instants clés (keyframes) et, pour les annotations, celles du jeu de validation. On garde donc seulement ces lignes ; les entrées du modèle sont inchangées.

```python
import json, shutil
D = '/content/data/nuscenes/v1.0-trainval'; B = D + '_full'
if not os.path.exists(B): shutil.copytree(D, B)
val = {i['scene_token'] for i in pickle.load(open('/content/data/infos/nuscenes_infos_temporal_val.pkl', 'rb'))['infos']}
vs = {s['token'] for s in json.load(open(f'{B}/sample.json')) if s['scene_token'] in val}
sd = [r for r in json.load(open(f'{B}/sample_data.json')) if r['is_key_frame']]
poses = {r['ego_pose_token'] for r in sd}
json.dump(sd, open(f'{D}/sample_data.json', 'w'))
json.dump([r for r in json.load(open(f'{B}/ego_pose.json')) if r['token'] in poses], open(f'{D}/ego_pose.json', 'w'))
json.dump([a for a in json.load(open(f'{B}/sample_annotation.json')) if a['sample_token'] in vs], open(f'{D}/sample_annotation.json', 'w'))
```

| Table | Original | Allégé |
|---|---|---|
| sample_data | 2 631 083 | 409 788 |
| ego_pose | 2 631 083 | 409 788 |
| sample_annotation | 1 166 187 | 192 041 |

Après cette cellule : **Runtime → Restart session**, pour libérer la RAM (les fichiers restent).

## 8. Lancer la tâche A

```python
import os
os.environ['CUDA_HOME'] = '/content/env'; os.environ['MPLBACKEND'] = 'Agg'
!cd /content/OpenDriveVLA && PYTHONPATH=/content/OpenDriveVLA:/content/OpenDriveVLA/drivevla \
  /content/env/bin/python -u /content/ai-safety-hw1-groupe4/code/run_opendrivevla.py \
  --targets 320 --save-images /content/results/images --out /content/results/A_sample320.json
```

## 9. (Facultatif) Archiver l'environnement sur Google Drive

```python
from google.colab import drive; drive.mount('/content/drive')
!tar -czf /content/drive/MyDrive/env_colab_complet.tar.gz -C /content env
```
