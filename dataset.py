import numpy as np
import collections
from torch.utils import *

import torchvision
from torchvision import *
from xml.etree.ElementTree import Element as ET_Element
import os
from typing import Any, Callable, Dict, Optional, Tuple, List
import torchvision.datasets.utils
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

import torch
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageList(object):
    def __init__(self, image_list, labels=None, transform=None):
        self.imgs = [((Image.open(open(val.split()[0], 'rb')).convert('RGB')), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)