import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from lucent.optvis import objectives, render, param
from src.analysis.dnn.utils import get_conv_layer_names, load_weights

"""
we generate maximally exciting images for all convolutional layers in
a given model instance. we define each *channel* in a layer to be a representational
unit. since we analyze the center pixel of every feature map, we maximize
activation for each center pixel across all channels
"""


def generate_meis(
    model: torch.nn.Module,
    savedir: str,
    image_size: int,
    layer_name: str,
    channel_index: int,
    device_id: Optional[str] = "0",
    decorrelate: Optional[bool] = True,
) -> None:
    """
    generate maximally exciting images for a given channel. the objective
    is to find an image which maximally excites the center pixel, since this is
    our `representational unit`.
    """
    # set everything to the correct GPU device
    device_id = f"cuda:{device_id}"
    model.to(device_id).eval()
    param_f = lambda: param.images.image(image_size, decorrelate=decorrelate)
    # define the layer + channel index for computing MEI
    obj = objectives.neuron(
        layer_name, channel_index
    )  # this defaults to the center pixel

    # compute image
    imgs = render.render_vis(model, obj, param_f=param_f, show_image=False)

    img = Image.fromarray((imgs[0][0] * 255).astype(np.uint8))

    # save image to disk
    savedir = os.path.join(savedir, layer_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    img.save(os.path.join(f"{savedir}", f"channel_{channel_index}_center.png"))


def main(
    model_type: str,
    base_path: str,
    savedir: str,
    seed: int,
    num_classes: int,
    image_size: int,
    device: str,
):
    ckpt_path = f"{base_path}/{model_type}/seed_{seed}/best.pth"
    model = load_weights(
        ckpt_path=ckpt_path,
        model_type=model_type,
        num_classes=num_classes,
    )

    all_conv_layers = get_conv_layer_names(model)

    # make layer names compatible with pytorch
    conv_layers_torch = [l.replace("_", ".") for l in all_conv_layers]

    # get the output channel dimension for every layer
    out_channels = []
    for name, module in model.named_modules():
        if name in conv_layers_torch:
            out_channels.append(module.out_channels)

    for idx, layer in tqdm(enumerate(all_conv_layers), total=len(all_conv_layers)):
        print(f"generating MEIs for layer: {layer}")
        for channel in tqdm(range(out_channels[idx])):
            generate_meis(
                model=model,
                savedir=savedir,
                image_size=image_size,
                layer_name=layer,
                channel_index=channel,
                device_id=device,
                decorrelate=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="model type for generating MEIs")
    parser.add_argument(
        "--num_classes", type=int, default=1000, help="number of classes in the model"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/chkapoor/pytorch-cifar/checkpoint_imagenet",
        help="base path for model checkpoints",
    )
    parser.add_argument("--seed", type=int, help="model seed to use")
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="spatial dimensions of generated MEI",
    )
    parser.add_argument("--savedir", type=str, help="directory path for saving MEIs")
    parser.add_argument("--device_id", type=int, help="cuda device ID")

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        base_path=args.base_path,
        savedir=args.savedir,
        seed=args.seed,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device_id,
    )
