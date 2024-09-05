from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim
import torch.optim.lr_scheduler
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Tester:

    '''
        General trainer class for single GPU or CPU training.
        An extensive list of parameters are needed to be supplied
        in the form of a dictionary. See example file in the /config folder
        at the project root.

        Imports are defined with the assumption that a main function operates
        in the parent folder of this class definition.
    '''

    def __init__(self, build_args: dict) -> None:

         self.build_args = build_args
         self.device = build_args["device"]

         self.model = self._build_model().to(self.device)

         self.loss_fn = getattr(torch.nn, build_args["loss_function"])()
         self.data_loader = self._build_data_loader()


    def _build_data_loader(self):
        data_set_name = self.build_args["data_set_name"]
        DataSet = getattr(import_module("data_sets"), data_set_name)
        data_set = DataSet(self.device, **self.build_args["data_set_args"])
        collate_fn = DataSet.collate_fn
        return DataLoader(data_set, **self.build_args["data_loader_args"],
                          collate_fn=collate_fn, )


    def _build_model(self):
        model_name = self.build_args["model_name"]
        Model = getattr(import_module(f"models.{model_name}"),
                        model_name.title())
        return Model(**self.build_args["model_args"])


    def _eval_one_image(self) -> float:
        # The data loader is assumed to handle device allocation from the
        # implemented pytorch DataSet.
        # Also, assumes that input data is a tuple and labels are torch tensors
        loss_sum = 0

        for i, data in enumerate(self.data_loader):
            input_data, label = data
            output = []
            with torch.no_grad():
                for chunk in self.build_args["test"]["chunks"]:
                    output += self.model(input_data).tolist()
            loss = self.loss_fn(output, label)

            loss_sum += loss.item()
            plt.imshow(np.array(output), interpolation='nearest')

            break

        return loss_sum / (i + 1)


    def run(self):
        log = self._eval_one_image()
        print(f"Loss: {log}")


