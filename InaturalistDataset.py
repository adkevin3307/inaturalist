import torch

import os
import json
import pandas as pd
from PIL import Image


class InaturalistDataset(torch.utils.data.Dataset):
    def __init__(self, label_file, root_dir, transform, category_filter=''):
        with open(label_file, 'r') as json_file:
            self.label_anns = json.load(json_file)

        self.label_file_df = pd.merge(
            pd.DataFrame(self.label_anns['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'}),
            pd.DataFrame(self.label_anns['annotations'])[['image_id', 'category_id']],
            on='image_id'
        )

        if isinstance(category_filter, list):
            self.label_file_df = self.label_file_df[self.label_file_df['category_id'].isin(category_filter)]
        elif isinstance(category_filter, str):
            self.label_file_df = self.label_file_df[self.label_file_df['file_name'].str.contains(category_filter)]

        for i, category in enumerate(sorted(set(self.label_file_df['category_id']))):
            self.label_file_df['category_id'] = self.label_file_df['category_id'].apply(lambda x: i if x == category else x)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_file_df)

    def __getitem__(self, idx):
        data = self.label_file_df.iloc[idx]

        image = Image.open(os.path.join(self.root_dir, data['file_name'])).convert('RGB')
        label = data['category_id']

        if self.transform:
            image = self.transform(image)

        return image, label

    def targets(self):
        return self.label_file_df['category_id'].nunique()
