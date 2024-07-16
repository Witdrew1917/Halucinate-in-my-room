import torch
import torch.optim
import torch.nn

from importlib import import_module

from tqdm import tqdm

class Trainer:

    def __init__(self, build_args: dict) -> None:

         self.build_args = build_args

         optimizer_fn = getattr(torch.optim, build_args["optimizer"])
         self.optimizer = optimizer_fn(**build_args["optimizer_args"])
         scheduler_fn = getattr(torch.optim, build_args["scheduler"])
         self.scheduler = scheduler_fn(**build_args["scheduler_args"])

         self.loss_fn = getattr(torch.nn, build_args["loss_function"])()
         self.learning_rate = build_args["learning_rate"]
         self.dropout = build_args["dropout"]

         self.data_loader = self._build_data_loader()
         self.model = self._build_model()


    def _build_data_loader(self):
        pass


    def _build_model(self):
        model_name = self.build_args["model_name"]
        Model = getattr(import_module(f"..models.{model_name}"), model_name)
        return Model(**self.build_args["model_args"])
    

