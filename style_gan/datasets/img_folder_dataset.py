from torch.utils.data import Dataset
import os
import numpy as np

from .pipelines import Compose
from .builder import DATASETS

@DATASETS.register_module()
class ImgFolderDataset(Dataset):

    IMG_EXTS = ['.jpg', '.png', '.bmp', '.gif', '.tiff', '.jpeg']
    
    def __init__(self, img_folder, pipeline):
        self.img_paths = self.get_img_paths(img_folder)
        self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        results = dict(img_path=img_path)
        return self.pipeline(results)

    def get_img_paths(self, img_folder):
        img_paths = list()
        for root, dirs, filenames in os.walk(img_folder):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.IMG_EXTS:
                    img_paths.append(os.path.join(root, filename))
        return img_paths