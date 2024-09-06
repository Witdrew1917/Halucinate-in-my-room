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
         self.test_args = build_args["test_args"]
         self.device = build_args["device"]

         self.model = self._build_model().to(self.device)

         self.loss_fn = getattr(torch.nn, build_args["loss_function"])()
         self.data_loader = self._build_data_loader()


    def _build_data_loader(self):
        data_set_name = self.build_args["data_set_name"]
        DataSet = getattr(import_module("data_sets"), data_set_name)
        data_set = DataSet(self.device, **self.test_args["data_set_args"])
        collate_fn = DataSet.collate_fn
        return DataLoader(data_set, **self.test_args["data_loader_args"],
                          collate_fn=collate_fn, )


    def _build_model(self):
        model_name = self.build_args["model_name"]
        Model = getattr(import_module(f"models.{model_name}"),
                        model_name.title())
        return Model(**self.build_args["model_args"])

    """
        I think this function kinda breaks my structure, but alas one should
        not worry to much about that. In the future this might be improvable.
    """
    def _eval_one_image(self) -> float:
        # The data loader is assumed to handle device allocation from the
        # implemented pytorch DataSet.
        # Also, assumes that input data is a tuple and labels are torch tensors
        loss_sum = 0

        for i, data in enumerate(self.data_loader):
            (input_pos, input_view), label = data
            output = []
            chunks = self.test_args["chunks"]
            with torch.no_grad():
                for chunk in tqdm(range(chunks)):
                    slice = self.test_args["data_set_args"]["rays_per_image"] \
                            / chunks
                    start = int(chunk*slice)
                    end = int((chunk+1)*slice)
                    output += self.model((input_pos[start:end,:],\
                            input_view[start:end,:])).tolist()

            image_width = self.test_args["image_width"]
            image_height = self.test_args["image_height"]

            image_array = np.array(output).reshape(image_height,image_width,3)

            _, ax = plt.subplots(1,2)
            ax[0].imshow(image_array.astype(int), interpolation='none')

            reference_image_array = \
                    label.numpy().reshape(image_height, image_width, 3)
            ax[1].imshow(reference_image_array.astype(int), interpolation='none')

            plt.show()
            break

        return loss_sum / (i + 1)*chunks


    def run(self):
        log = self._eval_one_image()
        print(f"Loss: {log}")


