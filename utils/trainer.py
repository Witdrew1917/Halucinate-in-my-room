from importlib import import_module
from time import perf_counter

import torch
import torch.optim
import torch.optim.lr_scheduler
import torch.nn
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

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

        self.jit = build_args["jit"]
        if build_args["jit"]:
            self._init_with_jax(build_args)
        else:
            self._init_with_torch(build_args)


    def _init_with_torch(self, build_args: dict) -> None:

         self.build_args = build_args
         self.verbose = build_args["verbose"]
         self.epochs = build_args["epochs"]
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


    def _init_with_jax(self, build_args: dict) -> None:

         self.build_args = build_args
         self.verbose = build_args["verbose"]
         self.epochs = build_args["epochs"]
         self.device = build_args["device"]

         optimizer_fn = getattr(optax, build_args["optimizer"])
         self.optimizer = optimizer_fn(**build_args["optimizer_args"])
         self.data_loader = self._build_data_loader()

         self.loss_fn = getattr(optax.losses, build_args["loss_function"])


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


    def _train_torch(self, iterator) -> float:
        # The data loader is assumed to handle device allocation from the
        # implemented pytorch DataSet.
        # Also, assumes that input data is a tuple and labels are torch tensors
        loss_list = []

        for _, data in enumerate(self.data_loader):

            t0 = perf_counter()

            input_data, label = data
            input_data = [torch.FloatTensor(data).to(self.device) \
                    for data in input_data]
            label = torch.FloatTensor(label).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_data)
            loss = self.loss_fn(output, label)
            loss.backward()

            self.optimizer.step()

            loss_list.append(loss.item())
            
            t1 = perf_counter()
            iterator.set_description(f"Loss: {loss.item()}, {t1-t0:.3f} sec")

        if self.scheduler:
            self.scheduler.step()

        return np.mean(loss_list).item()


    def save(self):
        return {'build_args': self.build_args,\
                'model_state_dict': self.model.state_dict(), \
                'ptimizer_state_dict': self.optimizer.state_dict()}


def run(trainer: Trainer, rng: int):
    iterator = tqdm(range(trainer.epochs), desc="Loss:")
    if not trainer.jit:
        for _ in iterator:
            log = trainer._train_torch(iterator)
            print(f"Loss: {log}")
    else:
        model = trainer._build_model()
        rng, init_rng = jax.random.split(jax.random.key(rng))
        params = model.init(init_rng, jnp.ones([1,3]), jnp.ones([1,3])\
                )['params']
        state = train_state.TrainState.create(apply_fn=model.apply, \
                params=params, tx=trainer.optimizer)

        for _ in iterator:
            log = train_jax(iterator, trainer, state)
            print(f"Loss: {log}")

        
@jax.jit
def apply_model(state, pos, view, labels):

    def aggregate(params):
        logits = state.apply_fn({'params': params}, pos, view)
        loss = jnp.mean(optax.losses.l2_loss(logits, labels))
        return loss, logits

    grad_fn = jax.value_and_grad(aggregate, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_jax(iterator, trainer: Trainer, state):
  
    loss_list = []
    for _, data in enumerate(trainer.data_loader):

        t0 = perf_counter()

        (pos, view), labels = data
        grads, loss = apply_model(state, pos, view, labels)
        state = update_model(state, grads)
        loss_list.append(loss)
        t1 = perf_counter()
        iterator.set_description(f"Loss: {loss}, {t1-t0:.3f} sec")

    return np.mean(loss_list).item()




