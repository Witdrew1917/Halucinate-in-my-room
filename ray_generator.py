import argparse
import os
import json
import sys
import pickle
from PIL import Image

FOCAL_LENGTH = 0.6911112070083618

def create_rays_synthetic_set(folder_path, transforms: dict):
    
    for frame in transforms["frames"]:

        file = frame["file_path"].split(".")[1]
        print(file)
        print(folder_path)

        with Image.open(folder_path + file + '.png') as img:
            pix = img.load()
            width = img.width
            height = img.height

            for y in range(height):
                for x in range(width):
                    if all(pix[x,y]):
                        print(pix[x,y])
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default='./data')
    parser.add_argument("--set_type", type=str, default='train')
    args = parser.parse_args()   


    if not args.set_type in ["train", "test", "val"]:
        print("Bad argument, target set must be named train, test or val.")
        sys.exit()


    with open(os.path.join(args.datafolder, \
            f"transforms_{args.set_type}.json")) as json_data:
        transforms = json.load(json_data)
        json_data.close()
        create_rays_synthetic_set(args.datafolder, transforms)
    











