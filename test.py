import argparse
import yaml

import torch
from utils.tester import Tester


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--recipe', type=str, required=True,
                        help="path to recipe file")
    parser.add_argument('-ld','--load_path', type=str, required=True,
                        help="full path (folder & file name) to where to saved model can be found")
    args = parser.parse_args()

    with open(args.recipe, 'r') as file:
        recipe = yaml.safe_load(file)

    model_name = [key for key in recipe.keys()][0]

    build_args = recipe[model_name]
    tester = Tester(build_args)
    tester.model.load_state_dict(torch.load(args.load_path)["model_state_dict"])

    tester.run()

if __name__ == '__main__':
    main()


