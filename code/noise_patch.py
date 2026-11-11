# Ajoute un bruit gaussien aux 6 caméras juste après le chargement des images (avant normalisation)
import os, numpy as np, cv2
from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadMultiViewImageFromFilesInCeph as Loader

def enable_from_env():
    sigma = float(os.environ.get('NOISE_SIGMA', '0'))
    seed  = int(os.environ.get('NOISE_SEED', '4'))
    save  = os.environ.get('NOISE_SAVE')
    state = {'calls': 0}
    orig = Loader.__call__

    def call(self, results):
        results = orig(self, results)
        imgs = results['img']
        if state['calls'] == 0:   # affiche l'échelle des pixels (à noter dans le rapport)
            print(f'[noise] {len(imgs)} images, forme {imgs[0].shape}, type {imgs[0].dtype}, '
                  f'min {imgs[0].min():.0f}, max {imgs[0].max():.0f}', flush=True)
        if sigma > 0:
            rng = np.random.default_rng(seed * 1000 + state['calls'])     # reproductible
            results['img'] = [np.clip(im.astype(np.float32) + rng.normal(0, sigma, im.shape), 0, 255).astype(im.dtype)
                              for im in imgs]                               # bornage dans [0, 255]
        if save:   # caméra avant (CAM_FRONT) du dernier instant chargé = l'échantillon cible
            cv2.imwrite(save, np.clip(results['img'][0], 0, 255).astype(np.uint8))
        state['calls'] += 1
        return results

    Loader.__call__ = call
    print(f'[noise] activé : sigma={sigma}, seed={seed}', flush=True)
