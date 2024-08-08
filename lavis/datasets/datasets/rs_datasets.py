import os
from collections import OrderedDict
import numpy as np
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import __DisplMixin
from PIL import Image


class RSCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_paths = [os.path.join(self.vis_root, image_path) for image_path in ann["images"]]
        images = [Image.open(image_path) for image_path in image_paths]
        bands = [image.split() for image in images]

        for i in range(3):
            assert len(bands[i]) == 3, "Image {} does not have three bands".format(i + 1)

        # Convert each band to numpy array and stack them along the depth (axis=2)
        bands_np = [np.array(band) for band in bands]

        # Stack along the new axis (0) to create a 9-band image and convert to tensor
        stacked_array = np.stack(bands_np, axis=0)
        stacked_tensor = torch.tensor(stacked_array, dtype=torch.float32).permute(1, 2, 0)

        image = self.vis_processor(stacked_tensor)
        caption = self.text_processor(ann["caption"][0])

        return {
            "image": image,
            "text_input": caption,
            # "image_id": self.img_ids[ann["image_id"]],
        }


class RSCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_paths = [os.path.join(self.vis_root, image_path) for image_path in ann["images"]]
        images = [Image.open(image_path) for image_path in image_paths]
        bands = [image.split() for image in images]

        for i in range(3):
            assert len(bands[i]) == 3, "Image {} does not have three bands".format(i + 1)

        # Convert each band to numpy array and stack them along the depth (axis=2)
        bands_np = [np.array(band) for band in bands]

        # Stack along the new axis (0) to create a 9-band image and convert to tensor
        stacked_array = np.stack(bands_np, axis=0)
        stacked_tensor = torch.tensor(stacked_array, dtype=torch.float32).permute(1, 2, 0)

        image = self.vis_processor(stacked_tensor)

        return {
            'image': image,
        }
