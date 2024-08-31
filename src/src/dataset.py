import h5py
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        '''
        root : data path
        input_key 
            - kspace(kspace)
            - image_input(image)
        target_key
            - image_label(image)
        forward 
            - True : test
            - False : train
        transform : toTensor & some augmentations
        '''
        
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.root = [Path(r) for r in root]
        # self.augmentor = augmentor

        if not forward:
            for r in self.root:
                image_files = list((r / "image").iterdir())
                for fname in sorted(image_files):
                    num_slices = self._get_metadata(fname)

                    self.image_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

        for r in self.root:
            kspace_files = list((r / "kspace").iterdir())
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
