import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnywhereDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnywhereDataset, self).__init__(**kwargs)

        addition_file_path = kwargs["add_file_path"]

        self.grasp_files = glob.glob(os.path.join(addition_file_path, 'positive_grasp', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'prompt', '*.pkl'))
        self.prompt_dir = os.path.join(file_path, 'prompt')
        self.instruction_dir = os.path.join(addition_file_path, 'grasp_instructions')
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.rgb_dir = os.path.join(file_path, 'image')
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        if kwargs["seen"]:
            with open(os.path.join('split/grasp-anything++/test/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)
            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything++/test/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))


        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        # mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.split('/')[-1]
        rgb_file = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    def get_prompts(self, idx):
        grasp_file = self.grasp_files[idx].split('/')[-1]
        prompt_file, obj_id, part_id = grasp_file.split('_')
        instruction_file = grasp_file.split('.')[0]
        instruction_file += '.pkl'
        instruction_file = os.path.join(self.instruction_dir, instruction_file)
        prompt_file += '.pkl'
        prompt_file = os.path.join(self.prompt_dir, prompt_file)
        obj_id = int(obj_id)
        part_id = int(part_id.split('.')[0])

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x
        
        # with open(instruction_file, 'rb') as f:
        #     instruction = pickle.load(f)

        return prompt, queries[obj_id]