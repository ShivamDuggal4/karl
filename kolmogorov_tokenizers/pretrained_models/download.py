import os
from tqdm import tqdm
import requests

pretrained_checkpoints = {
    'imagenet100': {
        "karl_small_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/60plmx9s1jsu5axqb1frj/karl_small_vqgan_quantized_latents.pth?rlkey=v0y3fsureyfijy5m28ba4vpl7&st=haisv8qx&dl=0",
        "karl_small_vqgan_continuous_latents.pth": "https://www.dropbox.com/scl/fi/ipvxil7ubb41wxs25du5p/karl_small_vqgan_continuous_latents.pth?rlkey=opmua15uarn2yt60yo9j7v5uo&st=bii053ux&dl=0",
        "karl_small_vae_quantized_latents.pth": "https://www.dropbox.com/scl/fi/gynbn6onb5tm1t8pmca0x/karl_small_vae_quantized_latents.pth?rlkey=93nz1j334ex57myea0qdxu7y5&st=9hjwvyc2&dl=0",
        "karl_small_vae_continuous_latents.pth": "https://www.dropbox.com/scl/fi/v5t6trgczpaprvoow58zk/karl_small_vae_continuous_latents.pth?rlkey=0tldlcy8ir0lrectdk1o42tch&st=qzfw7g1h&dl=0",
    },
    'imagenet':{
        "karl_small_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/4a2zwpceavl92ij56ja3m/karl_small_vqgan_quantized_latents.pth?rlkey=lc49wt2igsdhzfrqwvl62q30z&st=j3f9up9q&dl=0",
    }
    
}

def download_all(overwrite=False):
    base_download_path = "kolmogorov_tokenizers/pretrained_models/"
    for dataset in pretrained_checkpoints.keys():
        if not os.path.exists(os.path.join(base_download_path, dataset)):
            os.system('mkdir -p ' + os.path.join(base_download_path, dataset))
        for ckpt in pretrained_checkpoints[dataset].keys():
            download_path = os.path.join(base_download_path, dataset, ckpt)
            if not os.path.exists(download_path) or overwrite:
                headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
                r = requests.get(pretrained_checkpoints[dataset][ckpt], stream=True, headers=headers)
                print("Downloading {} | {} ...".format(dataset, ckpt))
                with open(download_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                        if chunk:
                            f.write(chunk)


if __name__ == "__main__":
    download_all()

