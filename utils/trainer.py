import torch
import torch.optim
import torch.nn

from importlib import import_module

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

         self.build_args = build_args
         self.device = build_args["device"]

         optimizer_fn = getattr(torch.optim, build_args["optimizer"])
         self.optimizer = optimizer_fn(**build_args["optimizer_args"])
         scheduler_fn = getattr(torch.optim, build_args["scheduler"])
         self.scheduler = scheduler_fn(**build_args["scheduler_args"])

         self.loss_fn = getattr(torch.nn, build_args["loss_function"])()
         self.learning_rate = build_args["learning_rate"]
         self.dropout = build_args["dropout"]

         self.data_loader = self._build_data_loader()
         self.model = self._build_model().to(self.device)



    def _build_data_loader(self):
        pass


    def _build_model(self):
        model_name = self.build_args["model_name"]
        Model = getattr(import_module(f"models.{model_name}"), model_name)
        return Model(**self.build_args["model_args"])


    @staticmethod
    def _call_model(input_data, model):
        '''
            This boiler plate function only exist such that models may
            override it if needed.
        '''
        print("Running default call to model")
        return model(*input_data)


    def _train_one_epoch(self) -> float:
        # The data loader is assumed to handle device allocation from the
        # implemented pytorch DataSet.
        # Also, assumes that input data is a tuple and labels are torch tensors
        loss_sum = 0

        for i, data in enumerate(self.data_loader):
            input_data, label = data
            self.optimizer.zero_grad()
            output = self._call_model(input_data, self.model)
            loss = self.loss_fn(output, label)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            loss_sum += loss.item()

        return loss_sum / i


    def run(self, epochs: int):
        for epoch in tqdm(range(epochs)):
            log = self._train_one_epoch()
            print(f"Epoch {epoch}: {log}")
