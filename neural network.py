import multiprocessing
import torch
from psutil import virtual_memory

ram_gb = round(virtual_memory().total / 1024**3, 1)

print('CPU:', multiprocessing.cpu_count())
print('RAM GB:', ram_gb)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device.type)

!nvidia-smi
from rudalle.pipelines import generate_images, show, convert_emoji_to_rgba, show_rgba, super_resolution
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_emojich_unet, get_realesrgan
from rudalle.utils import seed_everything
# prepare models:
device = 'cuda'
dalle = get_rudalle_model('Emojich', pretrained=True, fp16=True, device=device)
emojich_unet = get_emojich_unet('unet_effnetb5').to(device)
realesrgan = get_realesrgan('x4', device=device)
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)
text = 'Дональд Трамп из лего'

seed_everything(42)
pil_images = []
for top_k, top_p, images_num in [
    (2048, 0.995, 16),
]:
    pil_images += generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p, bs=8)[0]

show(pil_images, 4)

sr_images = super_resolution(pil_images, realesrgan)
rgba_images, _ = convert_emoji_to_rgba(sr_images, emojich_unet,  device=device)
for rgba_image in rgba_images:
    show_rgba(rgba_image);
____________________________________________________________________________
endln
