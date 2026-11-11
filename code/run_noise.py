"""
Lancer OpenDriveVLA-0.5B sur des echantillons nuScenes et recuperer la trajectoire predite.
HW1 AI Safety - Groupe 4 - membre 1.

Principe
- Les echantillons sont numerotes comme dans le code d'OpenDriveVLA : jeu de validation nuScenes
  (6019 echantillons) trie par horodatage. L'echantillon de la tache A est le n. 320 (= 80 x 4).
- Le modele garde une memoire des instants precedents d'une meme scene. Avant chaque echantillon
  cible, on fait donc passer les instants precedents de sa scene, dans l'ordre.
- Point d'entree pour les autres membres : `transform`, une fonction appliquee aux images juste
  apres leur chargement et avant la normalisation du modele. Elle recoit une image
  (numpy float32, BGR, valeurs 0-255, forme (900, 1600, 3)), le numero de l'instant et le numero
  de la camera, et doit renvoyer une image de meme forme. Par defaut : aucune transformation.

Utilisation en ligne de commande (sans transformation, tache A)
  python run_opendrivevla.py --targets 320 --out results/A_sample320.json

Utilisation depuis un autre script (exemple pour les taches C et D)
  from run_opendrivevla import Runner
  runner = Runner()                                   # charge le modele et les donnees (1-3 min)
  res = runner.run_sample(320)                        # sans transformation
  res = runner.run_sample(320, transform=ma_fonction) # avec la fonction du membre 2
"""
import argparse, copy, json, os, sys, time

import numpy as np

REPO = os.environ.get('OPENDRIVEVLA_REPO', '/content/OpenDriveVLA')
os.chdir(REPO)
sys.path[:0] = [REPO, os.path.join(REPO, 'drivevla')]

import torch
from mmengine import Config
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import DataArguments
from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
from utils.tensor_utils import move_data_to_device
from utils.trajectory_utils import retrieve_traj


class ImageHook:
    """Etape inseree dans le pipeline d'images : chargement -> [transform] -> normalisation."""
    def __init__(self):
        self.transform, self.frame, self.save_front = None, 0, None

    def __call__(self, results):
        if self.transform is not None:
            results['img'] = [np.asarray(self.transform(img, self.frame, cam), dtype=np.float32)
                              for cam, img in enumerate(results['img'])]
        if self.save_front:                       # camera avant telle que le modele la recoit
            import cv2
            os.makedirs(os.path.dirname(os.path.abspath(self.save_front)), exist_ok=True)
            cv2.imwrite(self.save_front, np.clip(results['img'][0], 0, 255).astype(np.uint8))
        return results


class Runner:
    def __init__(self, model_path='checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct'):
        t0 = time.time()
        disable_torch_init()
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            model_path, None, model_name="llava_qwen", device_map="cuda",
            multimodal=True, attn_implementation="sdpa",
            overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True})
        self.model.eval()
        cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
        data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
        self.dataset = LLaVANuScenesDataset(self.tokenizer, data_args, cfg.data.test,
                                            llava_test_mode=True, use_uniad_pth=False)
        self.hook = ImageHook()
        self.dataset.pipeline.transforms.insert(1, self.hook)
        self.collator = DataCollatorForLLaVANuScenesDataset(tokenizer=self.tokenizer, llava_test_mode=True)
        self.infos = self.dataset.data_infos
        self._cache = None                        # memoire (sans transformation) du dernier echantillon
        print(f"Modele et donnees charges en {time.time()-t0:.0f} s ({len(self.infos)} echantillons)", flush=True)

    # ---- memoire du modele (suivi d'objets et vue du dessus des instants precedents)
    def _trackers(self):
        return [m for m in self.model.modules()
                if hasattr(m, 'prev_frame_info') and hasattr(m, 'test_track_instances')]

    def _reset_memory(self):
        for m in self._trackers():
            m.test_track_instances, m.scene_token, m.prev_bev = None, None, None
            m.l2g_r_mat, m.l2g_t, m.timestamp = None, None, None
            m.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}

    def _save_memory(self):
        skip = {'_parameters', '_buffers', '_modules', '_non_persistent_buffers_set', 'training'}
        snap = []
        for m in self._trackers():
            state = {}
            for k, v in m.__dict__.items():
                if k in skip or ('hook' in k and k.startswith('_')):
                    continue
                try:
                    state[k] = copy.deepcopy(v)
                except Exception:
                    state[k] = v
            snap.append((m, state))
        return snap

    @staticmethod
    def _restore_memory(snap):
        for m, state in snap:
            for k, v in state.items():
                try:
                    m.__dict__[k] = copy.deepcopy(v)
                except Exception:
                    m.__dict__[k] = v

    # ---- inference
    def _forward(self, idx):
        data = move_data_to_device(self.collator([self.dataset[idx]]), 'cuda')
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = self.model.generate(data["input_ids"], uniad_data=data.get("uniad_data"),
                                      uniad_pth=data.get("uniad_pth"),
                                      qa_instance_ind=data.get("qa_instance_ind"),
                                      do_sample=False, temperature=0, max_new_tokens=512, num_beams=1)
        return data["question"], self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def previous_frames(self, idx):
        """Indices des instants precedents de la meme scene, dans l'ordre."""
        scene, f = self.infos[idx]['scene_token'], self.infos[idx]['frame_idx']
        return sorted((i for i, x in enumerate(self.infos) if x['scene_token'] == scene and x['frame_idx'] < f),
                      key=lambda i: self.infos[i]['frame_idx'])

    def run_sample(self, idx, transform=None, transform_previous_frames=False, save_front_image=None):
        """Fait tourner le modele sur l'echantillon idx et renvoie un dictionnaire de resultats.
        transform : fonction (image, instant, camera) -> image, appliquee a l'echantillon cible.
        transform_previous_frames : True pour l'appliquer aussi aux instants precedents (memoire).
        save_front_image : chemin .jpg pour enregistrer la camera avant de l'echantillon cible."""
        prev = self.previous_frames(idx)
        if not transform_previous_frames and self._cache is not None and self._cache[0] == idx:
            self._restore_memory(self._cache[1])  # memoire deja calculee pour cet echantillon
        else:
            self._reset_memory()
            for i in prev:
                self.hook.transform = transform if transform_previous_frames else None
                self.hook.frame = i
                self._forward(i)
            if not transform_previous_frames:
                self._cache = (idx, self._save_memory())
        self.hook.transform, self.hook.frame, self.hook.save_front = transform, idx, save_front_image
        t1 = time.time()
        try:
            question, answer = self._forward(idx)
        finally:
            self.hook.transform, self.hook.save_front = None, None
        try:
            traj = retrieve_traj(answer)
        except Exception:
            traj = []                             # reponse sans coordonnees lisibles
        return dict(index=idx, sample_token=self.infos[idx]['token'], scene_token=self.infos[idx]['scene_token'],
                    frame_idx=self.infos[idx]['frame_idx'], n_previous_frames=len(prev),
                    trajectory=traj, answer=answer, question=question, time_s=round(time.time() - t1, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=int, nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--save-images', default=None, help='dossier pour la camera avant de chaque echantillon')
    a = p.parse_args()
    runner = Runner()
    import noise_patch; noise_patch.enable_from_env()
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r['index'] for r in results}
    for t in a.targets:
        if t in done:
            continue
        img = os.path.join(a.save_images, f'sample{t}_CAM_FRONT.jpg') if a.save_images else None
        r = runner.run_sample(t, save_front_image=img)
        print(f"Echantillon {t} ({r['n_previous_frames']} instants precedents) -> {r['trajectory']}", flush=True)
        results.append(r)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(results, open(a.out, 'w'), indent=1)
    print(f"Resultats enregistres dans {a.out}")


if __name__ == '__main__':
    main()
