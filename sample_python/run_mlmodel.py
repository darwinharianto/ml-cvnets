from coremltools.models import MLModel
from options.opts import get_training_arguments
from utils.tensor_utils import image_size_from_opts
from PIL import Image
import copy
from typing import Optional

from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from data.datasets.segmentation.labelme_loader import COLOR_PALETTE
import glob
import os
from tqdm import tqdm
def read_and_process_image(opts, image_fname: str, *args, **kwargs):
    input_img = Image.open(image_fname).convert("RGB")
    input_pil = copy.deepcopy(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image while maitaining the aspect ratio
    res_h, res_w = image_size_from_opts(opts)

    input_img = F_vision.resize(
        input_img,
        size=(512, 512),
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    
    input_pil = F_vision.to_pil_image(input_tensor)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_pil, orig_h, orig_w


def main(args: Optional[list[str]] = None, **kwargs):
    opts = get_training_arguments(args=args)
    return opts


def draw_mask_color(mask: torch.Tensor):
    
    segmentationImage = Image.fromarray((mask.numpy()).astype(np.uint8), mode="P")
    segmentationImage.putpalette(COLOR_PALETTE)
    return segmentationImage

def merge_image_with_mask_horizontal(image1, image2):

    image2 = image2.convert('RGB')
    #resize, first image
    image1 = image1.resize((426, 240))
    image2 = image2.resize((426, 240))
    image1_size = image1.size

    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))

    return new_image

if __name__=="__main__":

    opts = main()
    images = glob.glob("/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/*.jpg")
    
    modelPath = "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/results_converted/coreml_models_res/SegEncoderDecoder.mlmodel"
    model = MLModel(modelPath)

    os.makedirs('results_mlmodel_images', exist_ok=True)
    for image in tqdm(images):
        
        input_tensor, input_pil, orig_h, orig_w = read_and_process_image(opts, image)
        predictions = model.predict({
            "input":  input_pil
        })

        im = draw_mask_color(torch.argmax(torch.tensor(predictions['var_1398']), dim=1)[0])

        merged = merge_image_with_mask_horizontal(input_pil, im)
        merged.save(f'results_mlmodel_images/{os.path.basename(image)}')