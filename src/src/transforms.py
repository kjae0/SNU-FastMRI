import numpy as np
import torch
import fastmri

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentor=None, crop=True, crop_by_width=True):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor
        self.crop = crop
        self.crop_by_width = crop_by_width
        
    def crop_kspace(self, kspace, crop_size):
        assert kspace.shape[-1] == 2
        
        height = kspace.shape[-3]
        width = kspace.shape[-2]
        
        cropped_kspace = fastmri.fft2c(fastmri.ifft2c(kspace)[:, (height - crop_size[0]) // 2 : crop_size[0] + (height - crop_size[0]) // 2, \
            (width - crop_size[1]) // 2 : crop_size[1] + (width - crop_size[1]) // 2, :])
        
        return cropped_kspace
        
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        if self.augmentor is not None:
            kspace_t = to_tensor(input)
            kspace_complex = torch.stack((kspace_t.real, kspace_t.imag), dim=-1)
            aug_kspace, aug_target = self.augmentor(kspace_complex, target_size=(target.shape[-2], target.shape[-1]))

            if aug_target is not None:


                kspace_real = aug_kspace[..., 0]
                kspace_imag = aug_kspace[..., 1]
                kspace = torch.complex(kspace_real, kspace_imag)

                # kspace = kspace * mask
                kspace = torch.stack((kspace.real, kspace.imag), dim=-1)

                if mask.shape[0] == kspace.shape[-2]:
                    mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
                else:
                    mask = torch.from_numpy(mask.reshape(1, kspace.shape[-3], 1, 1).astype(np.float32)).byte()
                    
                if self.crop:
                    if self.crop_by_width:
                        width = min(kspace.shape[-2], kspace.shape[-3])
                        crop_size = (width, width)
                    else:
                        crop_size = aug_target.shape
                    kspace = self.crop_kspace(kspace, crop_size)
                return mask, kspace, aug_target, maximum, fname, slice

        kspace = to_tensor(input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        if self.crop:
            if self.crop_by_width:
                crop_size = (kspace.shape[-2], kspace.shape[-2])
            else:
                crop_size = target.shape
            kspace = self.crop_kspace(kspace, crop_size)
        return mask, kspace, target, maximum, fname, slice
    