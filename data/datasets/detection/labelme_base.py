#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import os, glob
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from data.datasets import DATASET_REGISTRY
from data.datasets.detection.base_detection import BaseDetectionDataset
from data.transforms import image_pil as T
from data.transforms.common import Compose
from utils import logger
from labelme import LabelFile
from PIL import Image, ImageDraw
from data.datasets.segmentation.labelme_loader import LabelmeSegmenter, COLOR_PALETTE
import io

from shapely import Polygon, intersection

class Polygon(Polygon):

    @classmethod
    def from_labelme_rectangle(cls, labelmeAnnot:list[list[float]]):
        xVals = labelmeAnnot[0][0], labelmeAnnot[1][0] 
        yVals = labelmeAnnot[0][1], labelmeAnnot[1][1] 
        xmin, xmax = min(xVals), max(xVals)
        ymin, ymax = min(yVals), max(yVals)
        coords = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin))
        return cls(coords)
    
    def to_labelme_rectangle(self):
        assert len(self.coords) == 5, "Instance has more than distinct 4 points, not a valid rectangle"

        return [[self.coords[0][0], self.coords[0][1]], [self.coords[2][0], self.coords[2][1]]]

    @classmethod
    def from_labelme_polygon(cls, labelmeAnnot:list[list[float]]):
        coords = [tuple(polygon) for polygon in labelmeAnnot] + [tuple(labelmeAnnot[0])]
        return cls(coords)
    
    def to_labelme_polygon(self):
        return [[coords[0], coords[1]] for coords in self.coords][:-1]


@DATASET_REGISTRY.register(name="labelme", type="detection")
class LabelMeDetection(BaseDetectionDataset):
    """Base class for the LabelMe Dataset. Sub-classes should implement
    training and validation transform functions.

    Args:
        opts: command-line arguments

    .. note::
        This class implements basic functions (e.g., reading image and annotations), and does not implement
        training/validation transforms. Detector specific sub-classes should extend this class and implement those
        methods. See `coco_ssd.py` as an example for SSD.
    """

    def __init__(
        self,
        opts,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        split = "train" if self.is_training else "val"
        self.ann_files = glob.glob(os.path.join(
            self.root, "{}/*.json".format(split)
        ))

        # disable printing, so that pycocotools print statements are not printed
        logger.disable_printing()

        self.labelme = LabelFile()
        self.img_dir = os.path.join(self.root, "{}".format(split))

        self.background_idx = 0 if getattr(opts, "dataset.detection.no_background_id") else 1
        
        self.num_classes = 5 + self.background_idx # ground, washer, body, top, marker

        # enable printing
        logger.enable_printing()

        setattr(opts, "model.detection.n_classes", self.num_classes)

    def share_dataset_arguments(self) -> Dict[str, Any]:
        """Returns the number of classes in detection dataset along with super-class arguments."""
        share_dataset_specific_opts: Dict[str, Any] = super().share_dataset_arguments()
        share_dataset_specific_opts["model.detection.n_classes"] = self.num_classes
        return share_dataset_specific_opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != LabelMeDetection:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        # group.add_argument(
        #     "--dataset.detection.no-background-id",
        #     action="store_true",
        #     default=False,
        #     help="Do not include background id in detection class labels. Defaults to False.",
        # )
        return parser

    def _evaluation_transforms(
        self, size: tuple, *args, **kwargs
    ) -> T.BaseTransformation:
        """Evaluation or Inference transforms (Resize (Optional) --> Tensor).

        .. note::
            Resizing the input to the same resolution as the detector's input is not enabled by default.
            It can be enabled by passing **--evaluation.detection.resize-input-images** flag.

        """
        aug_list = []
        if getattr(self.opts, "evaluation.detection.resize_input_images"):
            aug_list.append(T.Resize(opts=self.opts, img_size=size))

        aug_list.append(T.ToTensor(opts=self.opts))
        return Compose(opts=self.opts, img_transforms=aug_list)

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
            output_data["targets"]["box_labels"]: Shape is [Num of boxes]
            output_data["targets"]["box_coordinates"]: Shape is [Num of boxes, 4]
            output_data["targets"]["image_id"]: Shape is [1]
            output_data["targets"]["image_width"]: Shape is [1]
            output_data["targets"]["image_height"]: Shape is [1]
        """

        crop_size_h, crop_size_w, img_index = sample_size_and_index

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        labelme = self.labelme
        labelme.load(self.ann_files[img_index])

        image, img_name = self.get_image(labelme=labelme)
        im_width, im_height = image.size

        boxes, labels, mask = self.get_boxes_and_labels(
            labelme=labelme,
            image_width=im_width,
            image_height=im_height,
            include_masks=True,
        )

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes,
            "mask": mask,
        }

        if transform_fn is not None:
            data = transform_fn(data)

        output_data = {
            "samples": {
                "image": data["image"],
            },
            "targets": {
                "box_labels": data["box_labels"],
                "box_coordinates": data["box_coordinates"],
                "mask": data["mask"],
                "image_id": torch.tensor(img_index),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data

    def __len__(self):
        return len(self.ann_files)
    
    def _crop_ground_with_bolt_and_rename(self, labelme:LabelFile):
        bolts = [Polygon.from_labelme_rectangle(shape['points']) for shape in labelme.shapes if shape['label'] == 'bolt']
        grounds = [Polygon.from_labelme_polygon(shape['points']) for shape in labelme.shapes if shape['label'] == 'ground']

        groundPolygons = []
        for bolt in bolts:
            intersections = [intersection(bolt, ground) for ground in grounds if type(intersection(bolt, ground)) == Polygon and not type(intersection(bolt, ground)).is_empty]
            if len(intersections) > 0:
                groundPolygons.append(intersections[0])


        return

    def get_boxes_and_labels(
        self,
        labelme: LabelFile,
        image_width: int,
        image_height: int,
        *args,
        include_masks=False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get the boxes and label information

        Args:
            image_width: Width of the image
            image_height: Height of the image
            include_masks: Return instance masks or not

        Returns:
            A tuple of length 3:
                * Numpy array containing bounding box information in xyxy format.
                    The shape of array is [Num_of_boxes, 4].
                * Numpy array containing labels for each of the box. The shape of array is [Num_of_boxes]
                * When include_masks is enabled, a numpy array of instance masks is returned. The shape of the
                    array is [Num_of_boxes, image_height, image_width]

        """

        # get candidate boxes from bolt annotations
        candidateBoxes = np.array(
            [
                self._points_to_xyxy(
                    points=shape['points'], 
                    image_width=image_width,
                    image_height=image_height, 
                ) 
                for shape in labelme.shapes if shape['label'] == 'bolt'
            
            ]
        )
        # generate semantic masks from data
        semanticMaskImage = self.segmentation_image(classNames=self.class_names(), colorPallete=COLOR_PALETTE)

        labels = []
        boxes = []
        masks = []

    
        # use candidate boxes to crop out each instances, if there is nothing in the crop out, ignore
        for box in candidateBoxes:
            candidateImage = semanticMaskImage.crop(box)

            blankImage = Image.new(mode="P", size=(image_width, image_height))
            blankImage.putpalette(COLOR_PALETTE)
            blankImage.paste(candidateImage, (box[0], box[1]))
            arrayVal = np.array(blankImage)

            uniqueValues = np.unique(arrayVal)
            if len(uniqueValues) == 1 and uniqueValues[0] == 0:
                continue

            # found box with input #TODO is it correct to add with background_idx?
            for uniqueValue in uniqueValues:
                if uniqueValue == 0:
                    continue
                
                labels.append(uniqueValue + self.background_idx)
                xVals = np.where(arrayVal==uniqueValue)[0]
                yVals = np.where(arrayVal==uniqueValue)[1]
                boxes.append(np.array([min(xVals), min(yVals), max(xVals), max(yVals)]))
                masks.append(arrayVal==uniqueValue)
        boxes = np.array(boxes).reshape((-1,4))
        labels = np.array(labels).reshape((-1,))
        masks = np.array(masks).reshape((-1,image_height, image_width)).astype(np.uint8)
        
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        
        boxes = boxes[keep].astype('float64')
        labels = labels[keep]
        masks = masks[keep]

        assert len(boxes) == len(labels) == len(masks)

        if include_masks:
            return boxes, labels, torch.from_numpy(masks)
        else:
            return boxes, labels, None

    def _points_to_xyxy(self, points:list[list[float]], image_width:int, image_height:int):
        xVals = [point[0] for point in points]
        yVals = [point[1] for point in points] 
        xmin, xmax = min(xVals), max(xVals)
        ymin, ymax = min(yVals), max(yVals)
        
        
        return [
            int(max(0, xmin)),
            int(max(0, ymin)),
            int(min(xmax, image_width)),
            int(min(ymax, image_height)),
        ]

    def get_image(self, labelme: LabelFile) -> Tuple:
        """Return the PIL image for a given image id"""
        image = Image.open(io.BytesIO(labelme.imageData))
        file_name = labelme.imagePath

        return image, file_name
    


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

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\n\t num_classes={self.num_classes}"

    @staticmethod
    def class_names() -> List[str]:
        """Name of the classes in the COCO dataset"""
        return [
            "background",
            "ground",
            "washer",
            "body",
            "top",
            "marker",
        ]



