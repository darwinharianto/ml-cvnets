import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import copy


import copy
import glob
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

from common import SUPPORTED_IMAGE_EXTNS
from cvnets import get_model
from data import create_test_loader
from engine.utils import autocast_fn
from metrics.confusion_mat import ConfusionMatrix
from options.opts import get_training_arguments
from utils import logger, resources
from utils.color_map import Colormap
from utils.common_utils import create_directories, device_setup
from utils.ddp_utils import is_master
from utils.download_utils import get_local_path
from utils.tensor_utils import image_size_from_opts
from utils.visualization_utils import convert_to_cityscape_format


def predict_and_save(
    opts,
    input_tensor: Tensor,
    file_name: str,
    orig_h: int,
    orig_w: int,
    model: nn.Module,
    target_mask: Optional[Tensor] = None,
    device: Optional = torch.device("cpu"),
    conf_mat: Optional[ConfusionMatrix] = None,
    color_map: List = None,
    orig_image: Optional[Image.Image] = None,
    adjust_label: Optional[int] = 0,
    is_cityscape: Optional[bool] = False,
    *args,
    **kwargs
) -> None:
    """Predict the segmentation mask and optionally save them"""

    mixed_precision_training = getattr(opts, "common.mixed_precision", False)
    mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype", "float16")

    output_stride = getattr(opts, "model.segmentation.output_stride", 16)
    if output_stride == 1:
        # we set it to 32 because most of the ImageNet models have 5 downsampling stages (2^5 = 32)
        output_stride = 32

    if orig_image is None:
        orig_image = F_vision.to_pil_image(input_tensor[0])

    curr_h, curr_w = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_h // output_stride) * output_stride
    new_w = (curr_w // output_stride) * output_stride

    if new_h != curr_h or new_w != curr_w:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(
            input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=True
        )

    file_name = file_name.split(os.sep)[-1].split(".")[0] + ".png"

    # move data to device
    input_tensor = input_tensor.to(device)
    if target_mask is not None:
        target_mask = target_mask.to(device)

    with autocast_fn(
        enabled=mixed_precision_training, amp_precision=mixed_precision_dtype
    ):
        # prediction
        pred = model(input_tensor, orig_size=(orig_h, orig_w))

    if isinstance(pred, Tuple) and len(pred) == 2:
        # when segmentation mask from decoder and auxiliary decoder are returned
        pred = pred[0]
    elif isinstance(pred, Tensor):
        pred = pred
    else:
        raise NotImplementedError(
            "Predicted must should be either an instance of Tensor or Tuple[Tensor, Tensor]"
        )

    num_classes = pred.shape[1]
    pred_mask = pred.argmax(1).squeeze(0)

    if target_mask is not None and conf_mat is not None:
        conf_mat.update(
            target=target_mask,
            prediction=pred,
        )

    save_dir = getattr(opts, "common.exp_loc", None)
    pred_mask = pred_mask + adjust_label
    if target_mask is not None:
        target_mask = target_mask + adjust_label

    draw_colored_masks(
        opts=opts,
        orig_image=orig_image,
        pred_mask=pred_mask,
        target_mask=target_mask,
        results_location=save_dir,
        color_map=color_map,
        file_name=file_name,
    )



    if getattr(opts, "evaluation.segmentation.save_masks", True):
        print("save masks")

        draw_binary_masks(
            opts=opts,
            pred_mask=pred_mask,
            file_name=file_name,
            is_cityscape=is_cityscape,
            results_location=save_dir,
        )

def read_and_process_image(opts, image_fname: str, *args, **kwargs):
    input_img = Image.open(image_fname).convert("RGB")
    input_pil = copy.deepcopy(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image while maitaining the aspect ratio
    res_h, res_w = image_size_from_opts(opts)

    input_img = F_vision.resize(
        input_img,
        size=min(res_h, res_w),
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_pil, orig_h, orig_w


def draw_binary_masks(
    opts,
    pred_mask: Tensor,
    file_name: str,
    results_location: str,
    is_cityscape: Optional[bool] = False,
) -> None:
    """Save masks whose values ranges between 0 and number_of_classes - 1"""
    print("draw binary")
    no_color_mask_dir = "{}/predictions_no_cmap".format(results_location)
    if not os.path.isdir(no_color_mask_dir):
        os.makedirs(no_color_mask_dir, exist_ok=True)
    no_color_mask_f_name = "{}/{}".format(no_color_mask_dir, file_name)

    if is_cityscape:
        # convert mask values to cityscapes format
        pred_mask = convert_to_cityscape_format(img=pred_mask)
    pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
    pred_mask_pil.save(no_color_mask_f_name)


def draw_colored_masks(
    opts,
    orig_image: Image.Image,
    pred_mask: Tensor,
    target_mask: Tensor,
    file_name: str,
    results_location: str,
    color_map: Optional[List] = None,
) -> None:
    """Apply color map to segmentation masks"""

    alpha = getattr(opts, "evaluation.segmentation.overlay_mask_weight", 0.5)
    save_overlay_rgb_pred = getattr(
        opts, "evaluation.segmentation.save_overlay_rgb_pred", False
    )

    if color_map is None:
        color_map = Colormap().get_color_map_list()

    # convert predicted tensor to PIL images, apply color map and save
    pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
    pred_mask_pil.putpalette(color_map)
    pred_mask_pil = pred_mask_pil.convert("RGB")
    pred_color_mask_dir = "{}/predictions_cmap".format(results_location)
    if not os.path.isdir(pred_color_mask_dir):
        os.makedirs(pred_color_mask_dir, exist_ok=True)
    color_mask_f_name = "{}/{}".format(pred_color_mask_dir, file_name)
    pred_mask_pil.save(color_mask_f_name)
    logger.log("Predicted mask is saved at: {}".format(color_mask_f_name))

    if target_mask is not None:
        # convert target tensor to PIL images, apply colormap, and save
        target_mask_pil = F_vision.to_pil_image(target_mask.byte())
        target_mask_pil.putpalette(color_map)
        target_mask_pil = target_mask_pil.convert("RGB")
        target_color_mask_dir = "{}/gt_cmap".format(results_location)
        if not os.path.isdir(target_color_mask_dir):
            os.makedirs(target_color_mask_dir, exist_ok=True)
        gt_color_mask_f_name = "{}/{}".format(target_color_mask_dir, file_name)
        target_mask_pil.save(gt_color_mask_f_name)
        logger.log("Target mask is saved at: {}".format(color_mask_f_name))

    if save_overlay_rgb_pred and orig_image is not None:
        # overlay predicted mask on top of original image and save

        if pred_mask_pil.size != orig_image.size:
            # resize if input image size is not the same as predicted mask.
            # this is likely in case of labeled datasets where we use transforms on the input image
            orig_image = F_vision.resize(
                orig_image,
                size=pred_mask_pil.size[::-1],
                interpolation=F_vision.InterpolationMode.BILINEAR,
            )

        overlayed_img = Image.blend(pred_mask_pil, orig_image, alpha=alpha)
        overlay_mask_dir = "{}/predictions_overlay".format(results_location)
        if not os.path.isdir(overlay_mask_dir):
            os.makedirs(overlay_mask_dir, exist_ok=True)
        overlay_mask_f_name = "{}/{}".format(overlay_mask_dir, file_name)
        overlayed_img.save(overlay_mask_f_name)
        logger.log(
            "RGB image blended with mask is saved at: {}".format(overlay_mask_f_name)
        )

        # save original image
        rgb_image_dir = "{}/rgb_images".format(results_location)
        if not os.path.isdir(rgb_image_dir):
            os.makedirs(rgb_image_dir, exist_ok=True)
        rgb_image_f_name = "{}/{}".format(rgb_image_dir, file_name)
        orig_image.save(rgb_image_f_name)
        logger.log("Original RGB image is saved at: {}".format(overlay_mask_f_name))
        print(rgb_image_f_name)


def main(args: Optional[List[str]] = None, **kwargs):
    opts = get_training_arguments(args=args)

    image = "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/val/202002_40_2022_12_26_14_10_55.jpg"
    image_fname = image.split(os.sep)[-1]

    input_tensor, input_pil, orig_h, orig_w = read_and_process_image(opts, image)

    device = getattr(opts, "dev.device", torch.device("cpu"))

    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)
    
    with torch.no_grad():
        predict_and_save(
            opts=opts,
            input_tensor=input_tensor,
            file_name=image_fname,
            orig_h=orig_h,
            orig_w=orig_w,
            model=model,
            target_mask=None,
            device=device,
            orig_image=input_pil,
        )

if __name__=="__main__":
    main()