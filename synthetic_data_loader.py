import pickle
import os
import random

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self, root_folder: str, device: str, rays_per_image=1) -> None:
        super().__init__()

        """
            Args:
                root_folder:
                    path to the root folder of the data set. The root 
                folder is assumed to be organized in samples of images.

                rays_per_image:
                    number of rays sampled from each image at each 
                iteration of the dataset. OBS! Increasing this size implicitly 
                increases the batch size.

        """

        self.root_folder = root_folder
        self.file_list = os.listdir(root_folder)
        self.rays_per_image = rays_per_image
        self.device = device


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):

        file_name = self.file_list[index]
        with open(os.path.join(self.root_folder, file_name), 'rb') as file:
            data_point = pickle.load(file)
            pos = []
            view = []
            tgt = []

            for ray in range(self.rays_per_image):
                i = random.randint(0,len(data_point))
                pos.append(torch.FloatTensor(data_point[i]["position"]))
                view.append(torch.FloatTensor(data_point[i]["direction"]))
                tgt.append(torch.FloatTensor(data_point[i]["target"]))

            pos = torch.stack(pos).to(self.device)
            view = torch.stack(view).to(self.device)
            tgt = torch.stack(tgt).to(self.device)

            return (pos, view), tgt
                

