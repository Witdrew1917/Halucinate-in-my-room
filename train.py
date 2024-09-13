import argparse
import yaml
import os
from datetime import datetime 

import torch
import random
from utils.trainer import Trainer, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--recipe', type=str, required=True,
                        help="path to recipe file")
    parser.add_argument('-sav','--save_path', type=str, required=True,
                        help="path to where to save resulting model")
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        default=False, help="adds extra logging during epochs")
    args = parser.parse_args()

    with open(args.recipe, 'r') as file:
        recipe = yaml.safe_load(file)

    model_name = [key for key in recipe.keys()][0]

    build_args = recipe[model_name]
    build_args["verbose"] = args.verbose
    trainer = Trainer(build_args)

    run(trainer, random.randint(0,1000))

    date = datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = os.path.join(args.save_path,f"{model_name}_{date}.pt")
    torch.save(trainer.save(), save_path)
    print(f"Done! Saved model at: {save_path}")


if __name__ == '__main__':
    main()
