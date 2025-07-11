import argparse
import datetime
import numpy as np
import os, cv2, subprocess, glob
from tqdm import tqdm
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import torchmetrics, lpips
import utils.misc as misc
import adaptive_tokenizers
from dreamsim import dreamsim

def get_args_parser():
    parser = argparse.ArgumentParser('KARL evaluation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--model', default='alit_small', type=str, metavar='MODEL', help='Name of model to evaluate')
    parser.add_argument('--input_size', default=256, type=int, help='images input size')
    parser.add_argument('--ckpt', type=str, help='checkpoint to load')

    # ALIT arguments
    parser.add_argument('--base_tokenizer', default="vqgan", type=str, help='Base 2D Tokenizer. Current options: VQGAN | VAE')
    parser.add_argument('--quantize_latent', action='store_true', help='Quantization of 1D latent tokens (before passing to decoder)')
    parser.set_defaults(pin_mem=False)

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--testing_custom_images', action='store_true', help='set to test on custom images')
    parser.set_defaults(pin_mem=False)
    return parser


def save_imgs(reconstructed_imgs, save_path):
    # batch_size=1
    reconstructed_imgs = np.clip(reconstructed_imgs[0].cpu().numpy().transpose([1, 2, 0]) * 255, 0, 255)
    reconstructed_imgs = reconstructed_imgs.astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(save_path, reconstructed_imgs)

def compute_reconstruction_fid(data_dir):
    fid_results = {}
    gt_folder = os.path.join(data_dir, "gt")
    for folder in sorted(os.listdir(data_dir)):
        if "reconstructed_imgs" not in folder: continue
        fid_compute_comamnd = "python -m pytorch_fid {} {}".format(os.path.join(data_dir, folder), gt_folder) #os.path.join(data_dir, folder.replace("reconstructed_imgs_256", "gt")))
        result = subprocess.check_output(fid_compute_comamnd, shell=True, text=True)

        # Extract the printed FID value from the result
        # Assuming the output contains "FID: " followed by the value
        fid_line = [line for line in result.splitlines() if "FID:" in line]
        if fid_line:
            fid_value = fid_line[0].split("FID:")[1].strip()
            print("Folder: {} | FID value: {}".format(folder, fid_value))
        else:
            assert(False)
        fid_results[folder] = fid_value


def compute_reconstruction_losses(data_dir):
    loglaplace_losses = {}
    loggaussian_losses = {}
    lpips_losses = {}
    ssim_scores = {}
    psnr_scores = {}
    dreamsim_scores = {}
    total_images = {}
    all_imgs = {}

    to_tensor = transforms.ToTensor()

    # Initialize metrics
    lpips_model = lpips.LPIPS(net='alex').eval().cuda()
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).cuda()
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device="cuda")

    for folder in sorted(os.listdir(data_dir)):
        if "reconstructed_imgs" not in folder:
            continue

        all_reconstructed_imgs = sorted(glob.glob(os.path.join(data_dir, folder, "*.png")))
        all_gt_imgs = [path.replace(folder, "gt") for path in all_reconstructed_imgs]

        if len(all_reconstructed_imgs) == 0:
            print(f"Folder {folder} has no images, skipping...")
            assert(False)

        total_loglaplace_loss, total_loggaussian_loss = 0, 0
        loglaplace_losses[folder] = []
        loggaussian_losses[folder] = []
        lpips_losses[folder] = []
        ssim_scores[folder] = []
        psnr_scores[folder] = []
        dreamsim_scores[folder] = []
        total_images[folder] = 0
        all_imgs[folder] = []
        

        for recon_img_path, gt_img_path in tqdm(zip(all_reconstructed_imgs, all_gt_imgs), desc=folder):
            img = Image.open(recon_img_path).convert("RGB")
            gt = Image.open(gt_img_path).convert("RGB")
            img_tensor = to_tensor(img).cuda()
            gt_tensor = to_tensor(gt).cuda()
            total_images[folder] += 1

            # Reconstruction losses
            loglaplace_loss = torch.abs(img_tensor - gt_tensor).mean()
            loggaussian_loss = torch.pow(img_tensor - gt_tensor, 2).mean()

            # Perceptual and metric losses
            lpips_loss = lpips_model(img_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
            ssim_val = ssim_metric(img_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
            psnr_val = psnr_metric(img_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
            dreamsim_val = dreamsim_model(img_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()

            # Accumulate
            total_loglaplace_loss += loglaplace_loss
            total_loggaussian_loss += loggaussian_loss
            loglaplace_losses[folder].append(loglaplace_loss.item())
            loggaussian_losses[folder].append(loggaussian_loss.item())
            lpips_losses[folder].append(lpips_loss)
            ssim_scores[folder].append(ssim_val)
            psnr_scores[folder].append(psnr_val)
            dreamsim_scores[folder].append(dreamsim_val)
            all_imgs[folder].append(recon_img_path)

        avg_loglaplace_loss = total_loglaplace_loss / len(all_reconstructed_imgs)
        avg_loggaussian_loss = total_loggaussian_loss / len(all_reconstructed_imgs)

        print(f"Folder: {folder} | images: {total_images[folder]} | Avg Log-Laplace Loss = {avg_loglaplace_loss.item():.4f} | "
              f"Avg Log-Gaussian Loss = {avg_loggaussian_loss.item():.4f} | "
              f"Avg LPIPS = {sum(lpips_losses[folder]) / len(lpips_losses[folder]):.4f} | "
              f"Avg SSIM = {sum(ssim_scores[folder]) / len(ssim_scores[folder]):.4f} | "
              f"Avg PSNR = {sum(psnr_scores[folder]) / len(psnr_scores[folder]):.4f}"
              f"DreamSim = {sum(dreamsim_scores[folder])/len(dreamsim_scores[folder]):.4f}")

    np.savez(os.path.join(data_dir, "reconstruction_metrics.npz"), 
        loglaplace=loglaplace_losses,
        loggaussian=loggaussian_losses,
        lpips=lpips_losses,
        ssim=ssim_scores,
        psnr=psnr_scores,
        dreamsim=dreamsim_scores,
        total_images=total_images,
        all_imgs=all_imgs)
    return loglaplace_losses, loggaussian_losses, lpips_losses, ssim_scores, psnr_scores, dreamsim_scores, total_images

def evaluate(args, model, data_loader, save_dir, desired_reconstruction_quality=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    print_freq = 1
    header = 'Validation'
    for data_iter_step, (input_samples, _, filename) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input_samples = input_samples.to(args.device, non_blocking=True)
        filename = filename[0].replace(".JPEG", ".png")
        
        
        with torch.no_grad():
            input_token_budget = [32,64,96,128,160,192,224,256]
            _, _, all_logs = model.encode(
                input_samples, input_token_budget=input_token_budget, desired_reconstruction_quality=desired_reconstruction_quality)

            for iter_logs_dict in evaluation_logs:
                for key in iter_logs_dict.keys():
                    if "reconstructed_imgs" not in key: continue
                    
                    save_path = os.path.join(save_dir, key)
                    if not os.path.exists(save_path): os.system('mkdir -p ' + save_path)
                    save_imgs(iter_logs_dict[key], os.path.join(save_path, "{}".format(filename)), image_active_token_count)
        
        save_path = os.path.join(save_dir, "gt")
        if not os.path.exists(save_path): os.system('mkdir -p ' + save_path)
        save_imgs(input_samples, os.path.join(save_path, "{}".format(filename)), image_active_token_count)
        

    if not args.testing_custom_images:
        compute_reconstruction_fid(data_dir=save_dir)
        compute_reconstruction_losses(data_dir=save_dir)


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Set device using CUDA_VISIBLE_DEVICES, otherwise this will always use gpu:0
    args.device = torch.device('cuda:0')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not args.testing_custom_images:
        # we used this code to evaluate on all datasets like IN100, IN-1K, COCO, WiT
        assert(os.path.exists(os.path.join(args.data_path, 'val')))
        transform_val = transforms.Compose([
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor()
        ])
        dataset_val = misc.ImageFolderWithFilenames(os.path.join(args.data_path, 'val'), transform=transform_val)
    else:
        transform_val = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor()
        ])
        dataset_val = misc.ImageFolderWithFilenames(os.path.join(args.data_path), transform=transform_val)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=None, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    base_tokenizer_args = {
        "id": args.base_tokenizer,
        "is_requires_grad": False
    }
    model = adaptive_tokenizers.__dict__[args.model](
        base_tokenizer_args=base_tokenizer_args, quantize_latent=args.quantize_latent, 
        train_stage="full_finetuning"
    )
    
    model.to(args.device)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['ema'], strict=False)
    print(sum(p.numel() for p in model.parameters()), "num params")
    print("Loaded...", checkpoint['epoch'])

    print(f"Start Evaluation")
    start_time = time.time()
    desired_reconstruction_quality = 0.0
    evaluate(args, model, data_loader_val, save_dir=os.path.join(args.log_dir, "evaluation_logs_{}".format(desired_reconstruction_quality)), desired_reconstruction_quality=desired_reconstruction_quality)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Validation time {}'.format(total_time_str))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
