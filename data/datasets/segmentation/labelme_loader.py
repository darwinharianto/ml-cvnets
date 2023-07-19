from labelme.label_file import LabelFile
from PIL import Image, ImageDraw
import io
import glob

import base64
import json
import os.path as osp
import os

from tqdm import tqdm
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
from labelme import __version__

COLOR_PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]

class LabelFileError(Exception):
    pass

class LabelmeSegmenter(LabelFile):

    def load(self, filename, imageDir=None):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warning(
                    "Loading JSON file ({}) of unknown version".format(
                        filename
                    )
                )
            elif version.split(".")[0] != __version__.split(".")[0]:
                logger.warning(
                    "This JSON file ({}) may be incompatible with "
                    "current labelme. version in file: {}, "
                    "current version: {}".format(
                        filename, version, __version__
                    )
                )

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename) if imageDir is None else imageDir, data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData


    def segmentation_image(self, classNames:list[str], colorPallete:list[int]):

        shapeWithKnownClasses = list(filter(lambda x: x['label'] in classNames, self.shapes))
        if len(shapeWithKnownClasses) == 0:
            return None

        shapeWithKnownClasses.sort(key=lambda x: classNames.index(x['label']))

        width, height = Image.open(io.BytesIO(self.imageData)).size

        segmentationImage = Image.new(mode="P", size=(width, height))
        segmentationImage.putpalette(colorPallete)

        draw = ImageDraw.Draw(segmentationImage)
        for shape in shapeWithKnownClasses:
            if shape['shape_type'] == 'polygon':
                draw.polygon([(int(val[0]), int(val[1])) for val in shape['points']], fill=classNames.index(shape['label']))
            else:
                raise TypeError('Annotation is not a polygon!')
            
        return segmentationImage
    

if __name__=="__main__":

    classNames = ['background', 'ground', 'washer', 'body', 'top', 'marker']

    files = glob.glob('/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/*.json')
    imageDir = '/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/2525_20220101-20230426_全支店'
    saveDir = "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/segmentation_image"
    files = [
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/202002_9_2022-03-11_16-31-35_9996.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/202002_6_2022_02_09_09_22_56.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/202002_27_2022_08_06_11_21_09.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/202002_20_2022-06-28_13-36-25_0478.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/202002_16_2022_05_21_19_36_27.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/train/201005_15_2023-03-28_14-29-01_5609.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/val/201005_1_2022_01_17_15_41_54.json",
        "/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/dataset/bolt/val/201005_13_2023-03-10_15-09-27_4213.json",
    ]
    print("saved")
    a = LabelmeSegmenter()
    for file in tqdm(files):
        a.load(file, imageDir=imageDir)
        image2 = a.segmentation_image(classNames, COLOR_PALETTE)
        image2 = image2.convert('RGB')
        image2 = image2.resize((426, 240))
        
        os.makedirs(saveDir, exist_ok=True)
        if image2 is not None:
            im = Image.open(io.BytesIO(a.imageData))
            image1 = im.resize((426, 240))

            image1_size = image1.size
            image2_size = image2.size
            new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
            new_image.paste(image1,(0,0))
            new_image.paste(image2,(image1_size[0],0))
            new_image.save(f'{saveDir}/{os.path.basename(a.imagePath)}.png')
            print("saved")