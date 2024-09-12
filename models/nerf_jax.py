from flax import linen as nn
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp



class Nerf(nn.Module):
    hidden_dim: int
    output_dim: int
    embedding_size_position: int
    embedding_size_view: int

    @nn.compact
    def __call__(self, pos, view):

        # pos and view are formated according to embedding size etc...

        """
            block 1
        """
        logits= nn.Dense(self.hidden_dim)(pos)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)

        
        """
            block 2
        """
        logits = nn.Dense(self.hidden_dim)(jnp.concatenate((pos,logits), axis=-1))
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim)(logits)
        logits= nn.relu(logits)
        logits= nn.Dense(self.hidden_dim+1)(logits)

        density = nn.relu(logits[:,:,1])


        """
            rbg decoder
        """
        color = nn.Dense(self.hidden_dim//2)(jnp.concatenate((logits[:,:,1:], \
                view), axis=-1))
        color = nn.relu(color)
        color = nn.Dense(self.output_dim)(color)
        color = nn.sigmoid(color)

        return color, density

if __name__ == '__main__':
    key1, key2, key3 = random.split(random.key(0), 3)
    pos = random.uniform(key1, (1,1,3))
    view = random.uniform(key2, (1,1,3))

    model = Nerf(
        hidden_dim=10,
        output_dim=3,
        embedding_size_position=4,
        embedding_size_view=10
            )

    params = model.init(key3, pos, view)
    y = model.apply(params, pos, view)
    print(y)


