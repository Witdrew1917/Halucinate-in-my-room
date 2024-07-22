import pickle
import os
import random

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self, device: str, root_folder: str, rays_per_image=1) -> None:
        super().__init__()

        """
            Args:
                root_folder:
                    path to the root folder of the data set. The root 
                folder is assumed to be organized in samples of images.

                rays_per_image:
                    number of rays sampled from each image at each 
                iteration of the dataset. OBS! Increasing this size implicitly 
                increases the batch size and is also faster than using the 
                latter. This size is also bounded by the number of pixels in
                each image. Rays are randomly sampled within each image.

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

            return pos, view, tgt[:,:3]

    @staticmethod
    def collate_fn(batch):

        P, V, T = [], [], []
        for item in batch:
            pos, view, tgt = item
            P.append(pos)
            V.append(view)
            T.append(tgt)
            
        return (torch.cat(P), torch.cat(V)), torch.cat(T)


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from time import time

    t0 = time()
    data_set = SyntheticDataset("../preprocessed_data", "cpu", rays_per_image=4)
    test_loader = DataLoader(data_set, batch_size=1, 
                             collate_fn=SyntheticDataset.collate_fn)

    for data in test_loader:
        pos, view, tgt = data

        break
    t1 = time()
    print(f"time: {t1-t0}s")

                

