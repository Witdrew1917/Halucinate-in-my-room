import torch
from torch import nn
import torch.nn.functional as f

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.curdir))

from utils.positional_encodings import fourier_encoding


class NERF(nn.Module):

    def __init__(self, input_dim_position: int, input_dim_view: int,
                 hidden_dim: int, output_dim: int, embedding_size_poisiton: int,
                 embedding_size_view: int):

        super().__init__()

        self.block1 = nn.Sequential(
                nn.Linear(input_dim_position*2*embedding_size_poisiton,
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
                    hidden_dim + input_dim_position*2*embedding_size_poisiton,
                          hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim+1),
                )

        self.rgb_decoder = nn.Sequential(
                nn.Linear(hidden_dim + input_dim_view*2*embedding_size_view,
                          hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid(),
                )

        self.embedding_size_position = embedding_size_poisiton
        self.embedding_size_view = embedding_size_view


    def forward(self, pos, view):
        emb_pos = fourier_encoding(pos, self.embedding_size_position)
        emb_view = fourier_encoding(view, self.embedding_size_view)

        print(emb_pos.shape)

        logits = self.block1(emb_pos)
        print(logits.shape)
        logits = self.block2(torch.concatenate((emb_pos, logits), dim=-1))
        density = f.relu(logits[:,1])
        color = self.rgb_decoder(
                torch.concatenate((logits[:,1:],emb_view), dim=-1))

        return density, color


if __name__ == '__main__':

    '''
        This is a test function for the above class defintion.
    '''

    pos = torch.rand(1,3)
    view = torch.rand(1,3)

    model = NERF(input_dim_position = 3, input_dim_view = 3, hidden_dim=10,
                 output_dim = 3, embedding_size_poisiton = 10,
                 embedding_size_view = 4)

    print(model(pos, view))
    
