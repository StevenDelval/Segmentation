import urllib.request
from tqdm import tqdm

def telecharger_fichier(url, destination):
    def barre_de_progression(block_num, block_size, total_size):
        progress = block_num * block_size / total_size * 100
        pbar.update(progress - pbar.n)

    with tqdm(unit='%', unit_scale=True, unit_divisor=1024, miniters=1, desc="Téléchargement", ncols=80) as pbar:
        urllib.request.urlretrieve(url, destination, reporthook=barre_de_progression)

    pbar.close()
    print("Le fichier a été téléchargé avec succès.")

lien = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
destination = "./sam_vit_b_01ec64.pth"
telecharger_fichier(lien, destination)
