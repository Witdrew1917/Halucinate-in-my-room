import random
from math import pi

from flax import linen as nn
from jax import random as jrandom, numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class Nerf_Jax(nn.Module):
    hidden_dim: int
    output_dim: int
    embedding_size_position: int
    embedding_size_view: int

    @nn.compact
    def __call__(self, pos, view):

        # pos and view are formated according to embedding size etc...
        pos, view = self._create_ray(pos, view)

        pos = self._fourier_encoding(pos, self.embedding_size_position)
        view = self._fourier_encoding(view, self.embedding_size_view)

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

        density = nn.relu(logits[:,:,0])


        """
            rbg decoder
        """
        color = nn.Dense(self.hidden_dim//2)(jnp.concatenate((logits[:,:,1:], \
                view), axis=-1))
        color = nn.relu(color)
        color = nn.Dense(self.output_dim)(color)
        color = nn.sigmoid(color)

        volume = self._volume_render(color, jnp.expand_dims(density,-1))

        return volume


    @staticmethod
    def _fourier_encoding(x: Array, N:int) -> Array:
        out = []
        for i in range(N):
            out.append(jnp.sin(2 ** i * pi * x))
            out.append(jnp.cos(2 ** i * pi * x))

        return jnp.concatenate(out, axis=-1)


    @staticmethod
    def _create_ray(pos: Array, view: Array, time_len=8, granularity=0.125):
        time_scale = jnp.expand_dims(jnp.arange(0,time_len,granularity), 1)

        pos = jnp.expand_dims(pos, -2) + jnp.expand_dims(view, -2) * time_scale
        view = jnp.repeat(jnp.expand_dims(view, -2), len(time_scale), axis=1)
        return pos, view


    @staticmethod
    def _volume_render(color_vec: ArrayLike, density: ArrayLike,\
            granularity=0.125):
        memoryless_transparency = granularity*density
        transparency = jnp.exp(-jnp.cumsum(memoryless_transparency, axis=-1))
        volume = jnp.sum(
                transparency*(1 - jnp.exp(-memoryless_transparency))*color_vec,
                axis=1)

        return volume 
                       

if __name__ == '__main__':
    key1, key2, key3 = jrandom.split(jrandom.key(random.randint(0,1000)), 3)
    pos = jrandom.uniform(key1, (1,3))
    view = jrandom.uniform(key2, (1,3))

    model = Nerf_Jax(
        hidden_dim=10,
        output_dim=3,
        embedding_size_position=10,
        embedding_size_view=4
            )

    params = model.init(key3, pos, view)
    y = model.apply(params, pos, view)
    print(y)


