import torch

def create_ray(pos: torch.Tensor, view: torch.Tensor, time_len=1,
               granularity=0.01):
    time_scale = torch.arange(0,time_len,granularity)
    pos = pos.unsqueeze(-1) + view.unsqueeze(-1) * time_scale
    view = view.unsqueeze(-1).expand(-1,len(time_scale))
    return pos, view


def volume_render(color_vec: torch.Tensor, density: torch.Tensor, 
                  granularity=0.01):
    memoryless_transparency = granularity*density
    transparency = torch.exp(-torch.cumsum(memoryless_transparency, dim=-1))
    volume = torch.sum(
            transparency*(1 - torch.exp(-memoryless_transparency))*color_vec)

    return volume 

