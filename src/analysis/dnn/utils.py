from tqdm import tqdm
from typing import Optional, List
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms, datasets

from lucent.modelzoo.util import get_model_layers


def compute_accuracy(
    model: nn.Module, dloader: torch.utils.data.DataLoader, device_id: int
) -> float:
    """
    helper function to compute model accuracy.
    a dataloader must also be provided
    """
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dloader, position=0, leave=True):
            inputs, targets = inputs.to(f"cuda:{device_id}"), targets.to(
                f"cuda:{device_id}"
            )
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return (100 * correct) / total


def imagenet_dataloader(
    dirpath: str, batch_size: int = 32, num_workers: int = 2
) -> torch.utils.data.DataLoader:
    """
    create a dataloader with imagenet specific normalizations
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dset = datasets.ImageFolder(
        dirpath,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return dloader


def clean_weight_dict(model_weights):
    """
    remove `module.` from model checkpoints so that there
    is no issue in loading the file into a model instance
    """
    clean_weights = OrderedDict()
    for k, v in model_weights.items():
        if "module." in k:
            clean_weights[k[7:]] = v
        else:
            clean_weights[k] = v

    return clean_weights


def init_model_instance(model_type: str, num_classes: Optional[int] = 10):
    """
    initialize a model instance
    """
    # use torchvision models if we are analyzing imagenet
    if model_type == "resnet18":
        if num_classes == 1000:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=False
            )
        else:
            raise NotImplementedError

    elif model_type == "resnet50":
        if num_classes == 1000:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=False
            )
        else:
            raise notimplementederror

    elif model_type in ["vgg16", "vgg19"]:
        vgg_name = model_type.upper()
        if num_classes == 1000:
            if vgg_name == "vgg16":
                model = torch.hub.load(
                    "pytorch/vision:v0.10.0", "vgg16_bn", pretrained=False
                )
            elif vgg_name == "vgg19":
                model = torch.hub.load(
                    "pytorch/vision:v0.10.0", "vgg19_bn", pretrained=False
                )

        else:
            raise NotImplementedError

    else:
        raise ValueError(f"incorrect model type: {model_type}")

    return model


def load_weights(
    ckpt_path: str,
    model_type: str,
    num_classes: Optional[int] = 10,
):
    """
    load model weights to model instance;
    return the model instance
    """
    ftype = ckpt_path.split(".")[-1]
    if ftype == "pth":
        model_weights = torch.load(ckpt_path)["net"]
    elif ftype == "pt":
        model_weights = torch.load(ckpt_path)["model"]
    else:
        raise ValueError(f"incorrect file type: {ftype}")

    clean_weights = clean_weight_dict(model_weights=model_weights)

    model = init_model_instance(model_type=model_type, num_classes=num_classes)
    model.eval()
    model.load_state_dict(clean_weights)

    return model


def get_conv_layer_names(model) -> List[str]:
    """
    get names of all convolutional layers in a model.
    names here are returned in accordance with lucent expects,
    ie: `.` replaced with `_`.
    """
    return [l for l in get_model_layers(model) if "conv" in l or "features" in l]


def get_layer_names(model):
    """
    recursively find all convolutional and fully-connected
    layers in a model
    """
    layers = []
    for n, lname in model.named_modules():
        if isinstance(lname, nn.Conv2d) or isinstance(lname, nn.Linear):
            if "shortcut" in n:
                continue
            layers.append(n)

    return layers


def get_activations(model, dloader, layer_name="linear", device_id="0"):
    activations = []
    model = model.to(f"cuda:{device_id}")

    def hook(module, input, output):
        if output.ndim == 4:
            hpixel, wpixel = output.shape[-2] // 2, output.shape[-1] // 2
            output = output[..., hpixel, wpixel]
        activations.append(output.detach().cpu())

    submodule = dict(model.named_modules()).get(layer_name)
    handle = submodule.register_forward_hook(hook)

    with torch.no_grad():
        for img, _ in tqdm(dloader):
            img = img.to(f"cuda:{device_id}")
            _ = model(img)
            img = img.detach().cpu()
            del img
            torch.cuda.empty_cache()

    handle.remove()

    # shift all activations to a different device for faster computation
    # and lesser memory footprint
    # todo: need to remove hardcoding of shifting cuda devices
    # lame strategy, but shift activations to the next serial gpu
    alt_device_id = int(device_id) + 1
    activations = torch.cat(activations, dim=0).to(f"cuda:{alt_device_id}")

    return activations
