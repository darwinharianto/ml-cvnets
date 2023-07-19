#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import List, Mapping, Tuple, Union
import glob
from PIL import Image, ImageDraw
import io
import numpy as np
from torch import Tensor

from data.datasets import DATASET_REGISTRY
from data.datasets.segmentation.base_segmentation import BaseImageSegmentationDataset
from data.datasets.segmentation.labelme_loader import LabelmeSegmenter, COLOR_PALETTE


@DATASET_REGISTRY.register(name="labelme", type="segmentation")
class LabelMeSegmentationDataset(BaseImageSegmentationDataset):
    """Dataset class for the labelme dataset

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        split = "train" if self.is_training else "val"
        self.ann_files = glob.glob(os.path.join(
            self.root, "{}/*.json".format(split)
        ))
        self.img_dir = os.path.join(self.root, "{}".format(split))
        self.split = split
        self.labelme = LabelmeSegmenter()

        self.ignore_label = 255
        self.background_idx = 0

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int], *args, **kwargs
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index. Returned sample is transformed
        into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and labels of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["targets"]["mask"]: Shape is [Height, Width]

        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        _transform = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        labelme = self.labelme
        labelme.load(self.ann_files[img_index], imageDir=self.img_dir)
        rgb_img = Image.open(io.BytesIO(labelme.imageData))
    
        mask = labelme.segmentation_image(classNames=self.class_names(), colorPallete=COLOR_PALETTE) # maybe can be changed with self.segmentation_image
        path = os.path.join(self.img_dir, labelme.imagePath)

        data = {"image": rgb_img, "mask": None if self.is_evaluation else mask}

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        output_data = {"samples": data["image"], "targets": data["mask"]}

        if self.is_evaluation:
            im_width, im_height = rgb_img.size
            img_name = path.replace("jpg", "png")
            mask = output_data.pop("targets")
            output_data["targets"] = {
                "mask": mask,
                "file_name": img_name,
                "im_width": im_width,
                "im_height": im_height,
            }

        return output_data
    

    def segmentation_image(self, classNames:list[str], colorPallete:list[int]):

        shapeWithKnownClasses = list(filter(lambda x: x['label'] in classNames, self.labelme.shapes))
        if len(shapeWithKnownClasses) == 0:
            return None

        shapeWithKnownClasses.sort(key=lambda x: classNames.index(x['label']))

        width, height = Image.open(io.BytesIO(self.labelme.imageData)).size

        segmentationImage = Image.new(mode="P", size=(width, height))
        segmentationImage.putpalette(colorPallete)

        draw = ImageDraw.Draw(segmentationImage)
        for shape in shapeWithKnownClasses:
            if shape['shape_type'] == 'polygon':
                draw.polygon([(int(val[0]), int(val[1])) for val in shape['points']], fill=classNames.index(shape['label']))
            else:
                raise TypeError('Annotation is not a polygon!')
            
        return segmentationImage

    def __len__(self) -> int:
        return len(self.ann_files)

    @staticmethod
    def class_names() -> List[str]:
        return [
            'background', 
            'ground', 
            'washer', 
            'body', 
            'top', 
            'marker'
            ]
    