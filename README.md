# AI Safety – Concepts and Techniques · Homework 1 · Groupe 4

Université d'Oldenburg, novembre 2026.

- **Partie 1** : test de robustesse du modèle de conduite autonome OpenDriveVLA-0.5B sur nuScenes (tâches A à E).
- **Partie 2** : analyse STPA d'un assistant vocal à IA générative dans une voiture. Elle se trouve dans le rapport, pas dans ce dépôt.

## Contenu du dépôt

```
code/
  run_opendrivevla.py     lancer le modèle sur un échantillon choisi (membre 1)
  constraints.txt         versions fixées des bibliothèques
  ...                     scripts des autres membres (bruit, 30 images, couverture)
notebooks/                notebook final de la partie 1, avec ses sorties
results/                  résultats (JSON, tableaux, figures)
docs/INSTALLATION_COLAB.md   installation complète, étape par étape
```

Ne sont **pas** inclus, car trop lourds ou soumis à licence : l'environnement Python, le modèle (Hugging Face, accès contrôlé) et les données nuScenes (licence nuScenes). `docs/INSTALLATION_COLAB.md` explique comment les obtenir.

## Environnement

- Google Colab, GPU Tesla T4 (16 Go), environ 12,7 Go de RAM.
- Environnement séparé : Python 3.10.22, CUDA 12.1.1, gcc 11.4, PyTorch 2.1.2+cu121.
- mmcv-full 1.7.2 et mmdet3d 1.0.0rc6 compilés ; transformers au commit `1c39974a` ; deepspeed 0.14.2.
- Modèle lancé en float16, attention `sdpa`, décodage glouton (`do_sample=False`, `temperature=0`, `num_beams=1`).

### Écarts par rapport au dépôt d'origine

- Versions fixées : `numpy==1.26.1`, `setuptools==69.5.1`, `yapf==0.40.1`, `opencv-python==4.9.0.80`, `matplotlib==3.8.4`, `shapely==1.8.5.post1`, plus la variable `MPLBACKEND=Agg`. `flash-attn` n'est pas installé.
- Une ligne modifiée : `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`, ligne 34 (`seaborn-whitegrid` → `seaborn-v0_8-whitegrid`).
- Métadonnées nuScenes allégées pour tenir dans la RAM : instants clés seulement, plus les annotations des scènes de validation. Le modèle ne lit jamais les données retirées (détails dans `docs/INSTALLATION_COLAB.md`).

## Échantillons et seeds

- Numérotation : jeu de validation nuScenes (6 019 échantillons) **trié par horodatage**, comme dans le code d'OpenDriveVLA, **en comptant à partir de 0**.
- Tâche A : 80 × 4 = **indice 320**. Jeton `f38ef51177c84c2a9fa32584b79e601e`, scène `scene-0276` (« Intersection, peds, cars, scooter, parked bicycle »). L'indice 319 (le 320ᵉ en comptant à partir de 1) a aussi été lancé.
- Tâche D : `random.Random(4).sample(range(6019), 30)`, soit :
  `162, 210, 482, 544, 738, 845, 869, 1269, 1350, 1414, 1584, 1756, 1818, 1933, 2132, 2144, 2226, 2266, 2370, 2484, 2538, 2951, 3244, 3289, 3922, 4262, 4396, 4500, 5248, 5908`.
- Le modèle garde une mémoire des instants précédents d'une même scène : avant chaque échantillon, on fait passer les instants précédents de sa scène, dans l'ordre.

## Résultat de la tâche A

Trajectoire prédite pour l'échantillon 320 : 6 points (x, y) en mètres, un toutes les 0,5 s sur 3 s, y étant la direction d'avancement.

```
[(-0.03, 0.84), (-0.1, 2.16), (-0.25, 3.79), (-0.45, 5.68), (-0.69, 7.74), (-0.94, 9.94)]
```

## Utiliser le script commun

```python
from run_opendrivevla import Runner
runner = Runner()                                    # charge le modèle et les données
r = runner.run_sample(320)                           # sans modification des images
r = runner.run_sample(320, transform=ma_fonction)    # ma_fonction(img, instant, camera) -> img
print(r['trajectory'])
```

`transform` reçoit chaque image (float32, BGR, 0-255, 900 × 1600 × 3) juste après son chargement et avant la normalisation du modèle. Par défaut, seul l'échantillon cible est transformé ; `transform_previous_frames=True` transforme aussi les instants précédents.

## Usage de l'IA générative

Le code assisté par IA est signalé en tête de chaque fichier concerné. La déclaration complète (outils, sections, cellules, vérifications) figure dans le rapport.

## Groupe

Groupe 4 : membre 1 (installation, tâche A, dépôt et notebook), membre 2 (tâches C et E), membre 3 (tâche D), membre 4 (tâche B, rapport).
