import torch

def create_ray(pos: torch.Tensor, view: torch.Tensor, time_len=1,
               granularity=0.01):
    time_scale = torch.arange(0,time_len,granularity).unsqueeze(-1)
    pos = pos.unsqueeze(-2) + view.unsqueeze(-2) * time_scale
    view = view.unsqueeze(-2).expand(-1,len(time_scale),-1)
    return pos, view


def volume_render(color_vec: torch.Tensor, density: torch.Tensor, 
                  granularity=0.01):
    memoryless_transparency = granularity*density
    transparency = torch.exp(-torch.cumsum(memoryless_transparency, dim=-1))
    volume = torch.sum(
            transparency*(1 - torch.exp(-memoryless_transparency))*color_vec,
            dim=1)

    return volume 


if __name__ == '__main__':

    # (B,R,D)
    x = torch.rand(2,3)
    d = torch.rand(2,3)

    pos_ray, view_ray = create_ray(x,d)
    print(pos_ray.shape)
    print(view_ray.shape)

    
    color_vec = torch.rand(2,10,3)
    density = torch.rand(2,10,1)

    render = volume_render(color_vec, density)
    print(render.shape)
