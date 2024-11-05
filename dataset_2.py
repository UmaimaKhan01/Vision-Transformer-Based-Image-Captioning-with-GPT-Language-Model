#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
from PIL import Image
from transformers import GPT2TokenizerFast
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

class ImageCaptionDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, sample['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = f"{sample['caption']}<|endoftext|>"
        input_ids = torch.tensor(tokenizer.encode(caption, truncation=True))
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  

        return image, input_ids, labels

def collate_fn(batch):
    images, input_ids, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return images, input_ids, labels

# data augmentation transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomBrightness(0.2),
    transforms.RandomContrast(0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ToTensor(),
])


train_dataset = ImageCaptionDataset(train_df, train_image_dir, train_transform)
valid_dataset = ImageCaptionDataset(valid_df, valid_image_dir, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)


# In[ ]:




