# @author Tasuku Miura

import os
import pickle
import pandas as pd
import numpy as np

from skimage import io, transform
from skimage.color import gray2rgb
import scipy.misc as spm

from torch.utils.data import Dataset, DataLoader


class IMDBFaces(Dataset):

    def __init__(self, pkl_file, root_dir, reduce_classes, transform=None):
        self._pkl_file = pkl_file
        self._root_dir = root_dir
        self._reduce_classes = reduce_classes
        self._transform = transform
        self._samples = self._extract_and_load()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path = self._samples['image_inputs'].iloc[idx]
        # Get image and resize as raw sizes not consistent.
        # Raw image is gray scale 0-1 float so need to convert.
        # Convert to int as PIL expects int.
        img = np.uint8(transform.resize(
            io.imread(path, flatten=1),
                (128, 128)
        ) * 255)
        # Convert to 3d, as model expects 3d.
        # img = np.asarray([img, img, img])
        img = gray2rgb(img)

        if self._transform is not None:
            img = self._transform(img)

        return {
            'faces': img,
            'ages': self._samples['age_labels'].iloc[idx]
        }

    def _extract_and_load(self):
        pkl_path = os.path.join(self._root_dir, self._pkl_file)
        with open(pkl_path, 'rb') as f:
            pdict = pickle.load(f)
        # Convert to pandas data frames.
        img_df = pd.DataFrame(pdict['image_inputs'], columns=['image_inputs'])
        labels_df = pd.DataFrame(pdict['age_labels'], columns=['age_labels'])
        # Reduce the number of classes to simplify if _reduce_classes is true.
        if self._reduce_classes:
            labels_df = labels_df.age_labels.apply(self._classify_age)

        df = pd.concat([img_df, labels_df], axis=1)
        return df

    def _classify_age(self, x):
        if x < 30:
            return 0
        elif x >= 30 and x < 45:
            return 1
        elif x >= 45 and x < 60:
            return 2
        else:
            return 3
