import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tfs
from torchvision import datasets
from PIL import Image
from glob import glob
from random import randint, uniform, choice


class CutPaste(object):
    def crop_and_paste(self, image, crop_width, crop_height):
        org_width, org_height = image.size
        patch_left, patch_top = randint(0, org_width - crop_width), randint(
            0, org_height - crop_height
        )
        patch_right, patch_bottom = patch_left + crop_width, patch_top + crop_height
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))

        paste_left, paste_top = randint(0, org_width - crop_width), randint(
            0, org_height - crop_height
        )
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top))
        return aug_image

    def __call__(self, img, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1), (1, 3.3))):
        img_area = img.size[0] * img.size[1]
        patch_area = uniform(*area_ratio) * img_area
        patch_aspect = choice([uniform(*aspect_ratio[0]), uniform(*aspect_ratio[1])])
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        cutpaste_img = self.crop_and_paste(img, patch_w, patch_h)
        return img, cutpaste_img

    def __repr__(self):
        return self.__class__.__name__ + "()"


class CutPasteDataset(Dataset):
    def __init__(self, dataset_type, dataset_class, input, mode):
        self.transform = tfs.Compose(
            [
                tfs.Resize(256),
                tfs.CenterCrop(224),
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cutpaste = CutPaste()
        self.mode = mode
        self.path = os.path.join(os.getcwd(), 'datasets/data/', dataset_type, dataset_class, mode)
        self.images = []
        if self.mode == 'train':
            self.temp = glob(self.path + "/*/*")
            for image in self.temp:
                image = Image.open(image).convert('RGB')
                cutpaste_img = self.cutpaste(image)
                image = [self.transform(img) for img in cutpaste_img]
                self.images.append(image)  
        elif self.mode == 'test':
            self.images = datasets.ImageFolder(self.path, transform=self.transform)
            if dataset_type == 'mpdd2':
                self.images.samples = [(d, 0) if i == self.images.class_to_idx['normal'] else (d, 1) for d, i in self.images.samples]
            else:
                self.images.samples = [(d, 0) if i == self.images.class_to_idx['good'] else (d, 1) for d, i in self.images.samples]
        temp = 1    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mode == "train":
            out = self.images[index]
            return out
        elif self.mode == "test":
            path, label = self.images.samples[index]
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label