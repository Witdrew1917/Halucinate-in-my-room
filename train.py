import argparse
import yaml
import sys

import torch
from utils.trainer import Trainer
from utils.ray_utils import create_ray, volume_render


def nerf(input_data:tuple ,model):

    pos, view = input_data
    pos, view = create_ray(pos, view)

    color, density = model(pos, view)
    volume = volume_render(color, density)

    return volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--recipe', type=str, required=True,
                        help="path to recipe file")
    args = parser.parse_args()

    with open(args.recipe, 'r') as file:
        recipe = yaml.safe_load(file)

    model_name = [key for key in recipe.keys()][0]

    build_args = recipe[model_name]
    trainer = Trainer(build_args)
    trainer._call_model = getattr(sys.modules[__name__], model_name)
    trainer.run()


if __name__ == '__main__':
    main()
