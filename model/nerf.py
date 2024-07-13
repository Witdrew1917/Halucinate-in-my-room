import torch
from torch import nn


class NERF:

    def __init__(self, input_dim_position: int, input_dim_view: int,
                 hidden_dim: int, output_dim: int, embedding_size_poisiton: int,
                 embedding_size_view: int) -> None:

        self.block1 = nn.Sequential(
                nn.Linear(input_dim_position*embedding_size_poisiton,
                          hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                )

        self.block2 = nn.Sequential(
                nn.Linear(
                    hidden_dim + input_dim_position*embedding_size_poisiton,
                          hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                )

        self.density_decoder = nn.Linear(hidden_dim, hidden_dim+1)

        self.rgb_decoder = nn.Sequential(
                nn.Linear(hidden_dim + input_dim_view*embedding_size_view,
                          hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid(),
                )


    def forward(self, o, d):
        pass

        
