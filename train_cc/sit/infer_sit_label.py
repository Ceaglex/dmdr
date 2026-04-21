import argparse
import math
import os
from typing import List

import torch
from PIL import Image
from torchvision.utils import make_grid

from diffusers.models import AutoencoderKL

from models_sit import SiT_models
from samplers import euler_sampler, v2x0_sampler


def parse_labels(labels: str) -> List[int]:
    out = []
    for token in labels.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("--labels is empty. Example: --labels 0,1,2,3")
    return out


def array2grid(x: torch.Tensor) -> Image.Image:
    nrow = max(1, round(math.sqrt(x.size(0))))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(x)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SiT label-conditional inference")
    parser.add_argument("--model", type=str, default="SiT-XL/2", help="Model key in SiT_models")
    parser.add_argument("--pretrain-path", type=str, required=True, help="Path to converted SiT checkpoint")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema", help="HF repo id or local dir")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated class ids, e.g. 0,1,2,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples-per-label", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=8, help="Sampling steps")
    parser.add_argument("--sampler", type=str, choices=["euler", "v2x0"], default="v2x0")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Used by euler sampler")
    parser.add_argument("--shift", type=float, default=1.0, help="Used by v2x0 sampler")
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--out", type=str, default="infer_sit_label.png")
    return parser


def main():
    args = build_argparser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    torch.manual_seed(args.seed)

    assert args.resolution % 8 == 0, "resolution must be divisible by 8"
    latent_size = args.resolution // 8

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=1000,
        use_cfg=False,
        fused_attn=True,
        qk_norm=False,
    )
    state_dict = torch.load(args.pretrain_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device=device, dtype=dtype)
    vae.eval()

    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215], device=device, dtype=dtype).view(1, 4, 1, 1)
    latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 4, 1, 1)

    labels = parse_labels(args.labels)
    all_labels = []
    for y in labels:
        if y < 0 or y > 999:
            raise ValueError(f"Label must be in [0, 999], got: {y}")
        all_labels.extend([y] * args.num_samples_per_label)

    y = torch.tensor(all_labels, device=device, dtype=torch.long)
    z = torch.randn((len(all_labels), 4, latent_size, latent_size), device=device, dtype=dtype)

    with torch.no_grad():
        if args.sampler == "euler":
            latents = euler_sampler(
                model,
                z,
                y,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
            )
        else:
            latents, _, _ = v2x0_sampler(
                model,
                z,
                y,
                num_steps=args.num_steps,
                shift=args.shift,
            )

        images = vae.decode((latents - latents_bias) / latents_scale).sample
        images = ((images + 1) / 2).clamp(0, 1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    grid = array2grid(images.float())
    grid.save(args.out)
    print(f"Saved image grid to: {args.out}")
    print(f"Labels: {all_labels}")


if __name__ == "__main__":
    main()
