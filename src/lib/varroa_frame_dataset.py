import torch
import torchvision
import os
import torchvision.transforms.v2
from pathlib import Path
from PIL import Image

class VarroaDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path: str, image_processing:torchvision.transforms.v2.Compose, balance:bool=False):
        self.folder_path = folder_path
        self.frames = os.listdir(self.folder_path)
        self.image_processing = image_processing
        if balance:
            self.balance()

        print(f"Loaded from {folder_path} a total of {len(self.frames)} images, 1: {self.varroa_infested_count()}, 0: {self.varroa_free_count()}")

    def varroa_free_count(self):
        return sum(list(map(lambda x:not self.get_label_by_frame(x), self.frames)))
    
    def varroa_infested_count(self):
        return sum(list(map(lambda x:self.get_label_by_frame(x), self.frames)))
    
    def balance(self): #Remove elements from the lower dataset to balance the dataset
        while self.varroa_free_count() > self.varroa_infested_count():
            for index, item in enumerate(self.frames):
                if self.get_label_by_frame(item)==0:
                    self.frames.pop(index)
                    break
        while self.varroa_free_count() < self.varroa_infested_count():
            for index, item in enumerate(self.frames):
                if self.get_label_by_frame(item)==1:
                    self.frames.pop(index)
                    break


    def __len__(self):
        return len(self.frames)
    
    def get_label_by_frame(self, path:str) -> int:
        return int(Path(path).stem.split("_")[-1])
    
    def __getitem__(self, index):
        frame_path = self.frames[index]
        label = self.get_label_by_frame(frame_path)
        absolute_path = os.path.join(self.folder_path, frame_path)
        img = torchvision.io.decode_image(absolute_path)
        img = self.image_processing(img)
        return img, label

