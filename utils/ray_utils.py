import torch

def create_ray(pos: torch.Tensor, view: torch.Tensor, granularity=0.01):
    time_scale = torch.arange(0,2,granularity)
    pos = pos.unsqueeze(-1) + view.unsqueeze(-1) * time_scale
    view = view.unsqueeze(-1).expand(-1,len(time_scale))
    return pos, view

