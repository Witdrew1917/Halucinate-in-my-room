from importlib import import_module
from time import perf_counter

import torch
import torch.optim
import torch.optim.lr_scheduler
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    '''
        General trainer class for single GPU or CPU training.
        An extensive list of parameters are needed to be supplied
        in the form of a dictionary. See example file in the /config folder
        at the project root.

        Imports are defined with the assumption that a main function operates
        in the parent folder of this class definition.
    '''

    def __init__(self, build_args: dict) -> None:

         self.verbose = build_args["verbose"]
         self.epochs = build_args["epochs"]
         self.build_args = build_args
         self.device = build_args["device"]

         self.model = self._build_model().to(self.device)
         optimizer_fn = getattr(torch.optim, build_args["optimizer"])
         self.optimizer = optimizer_fn(self.model.parameters(),
                                       **build_args["optimizer_args"])
         scheduler_fn = getattr(torch.optim.lr_scheduler,
                                build_args["scheduler"])
         self.scheduler = scheduler_fn(self.optimizer, 
                                       **build_args["scheduler_args"])

         self.loss_fn = getattr(torch.nn, build_args["loss_function"])()
         self.dropout = build_args["dropout"]

         self.data_loader = self._build_data_loader()



    def _build_data_loader(self):
        data_set_name = self.build_args["data_set_name"]
        DataSet = getattr(import_module("data_sets"), data_set_name)
        data_set = DataSet(**self.build_args["data_set_args"])
        collate_fn = DataSet.collate_fn
        return DataLoader(data_set, **self.build_args["data_loader_args"],
                          collate_fn=collate_fn)


    def _build_model(self):
        model_name = self.build_args["model_name"]
        Model = getattr(import_module(f"models.{model_name}"),
                        model_name.title())
        return Model(**self.build_args["model_args"])


    def _train_one_epoch(self, iterator) -> float:
        # The data loader is assumed to handle device allocation from the
        # implemented pytorch DataSet.
        # Also, assumes that input data is a tuple and labels are torch tensors
        loss_sum = 0
        t0 = perf_counter()

        for i, data in enumerate(self.data_loader):


            input_data, label = data
            
            input_data = [torch.FloatTensor(data).to(self.device) \
                    for data in input_data]
            label = torch.FloatTensor(label).to(self.device)

            t1 = perf_counter()
            self.optimizer.zero_grad()
            output = self.model(input_data)
            loss = self.loss_fn(output, label)
            loss.backward()

            self.optimizer.step()

            loss_sum += loss.item()
            iterator.set_description(f"Loss: {loss.item()}")

            if i==0:
                break

        if self.scheduler:
            self.scheduler.step()

        t2 = perf_counter()

        if self.verbose:
            print(f"_train_one_epoch executed one iteration after avg {(t2 - t0) / (i + 1)} seconds")
            print(f"of this time {t1-t0} was spent loading the data")

        return loss_sum / (i + 1)


    def run(self):
        iterator = tqdm(range(self.epochs), desc="Loss:")
        for _ in iterator:
            log = self._train_one_epoch(iterator)
            print(f"Loss: {log}")

    def save(self):
        return {'build_args': self.build_args,\
                'model_state_dict': self.model.state_dict(), \
                'ptimizer_state_dict': self.optimizer.state_dict()}

