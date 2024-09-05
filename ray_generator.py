import argparse
import json
import os
import pickle
import sys
from PIL import Image

#third party
import numpy as np
from tqdm import tqdm


FOCAL_LENGTH = 0.6911112070083618


def virtual_camera(pixel_width: int, pixel_height:int) -> np.ndarray:
    origin = np.array([pixel_width/2,pixel_height/2, FOCAL_LENGTH])
    X, Y = np.meshgrid(np.arange(pixel_width), np.arange(pixel_height))
    camera_matrix = np.stack((X, Y, np.zeros(X.shape)), axis=2) \
            - origin.reshape(1,1,3)

    norm = np.linalg.norm(camera_matrix, axis=1, keepdims=True)
    tol = 1e-03
    norm[norm < tol] = tol

    return camera_matrix / norm


def create_rays_synthetic_set(folder_path, frame: dict):
    
    data_set = []
    file = frame["file_path"].split(".")[1]

    with Image.open(folder_path + file + '.png') as img:
        pix = img.load()
        width = img.width
        height = img.height

        camera = virtual_camera(width, height)
        rotation_matrix = np.array(frame["transform_matrix"])
        ray_matrix = np.matmul(camera, rotation_matrix[:3,:3])


        for y in range(height):
            for x in range(width):

                sample = {'position': rotation_matrix[:3,3].tolist(),
                          'direction': ray_matrix[x,y,:].tolist(),
                          'target':pix[x,y]}

                data_set.append(sample)

    return data_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default='./data')
    parser.add_argument("--set_type", type=str, default='train')
    args = parser.parse_args()   

    output_dir = './preprocessed_data'

    if not args.set_type in ["train", "test", "val"]:
        print("Bad argument, target set must be named train, test or val.")
        sys.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(args.datafolder, \
            f"transforms_{args.set_type}.json")) as json_data:
        transforms = json.load(json_data)
        json_data.close()

        for image_id, frame in enumerate(tqdm(transforms["frames"])):
            data_set = create_rays_synthetic_set(args.datafolder, frame)
            with open(f"{output_dir}/{args.set_type}_{image_id}.pkl", 'wb') as file:
                pickle.dump(data_set, file=file)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default='./data')
    parser.add_argument("--set_type", type=str, default='train')
    args = parser.parse_args()   

    output_dir = './preprocessed_data'

    if not args.set_type in ["train", "test", "val"]:
        print("Bad argument, target set must be named train, test or val.")
        sys.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(args.datafolder, \
            f"transforms_{args.set_type}.json")) as json_data:
        transforms = json.load(json_data)
        json_data.close()

        for image_id, frame in enumerate(tqdm(transforms["frames"])):
            data_set = create_rays_synthetic_set(args.datafolder, frame)
            with open(f"{output_dir}/{args.set_type}_{image_id}.pkl", 'wb') as file:
                pickle.dump(data_set, file=file)

