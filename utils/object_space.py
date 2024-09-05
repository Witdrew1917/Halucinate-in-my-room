import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from itertools import product

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default='../data')
    parser.add_argument("--set_type", type=str, default='train')
    args = parser.parse_args()   

    if not args.set_type in ["train", "test", "val"]:
        print("Bad argument, target set must be named train, test or val.")
        sys.exit()

    with open(os.path.join(args.datafolder, \
            f"transforms_{args.set_type}.json")) as json_data:
        transforms = json.load(json_data)
        json_data.close()

        P = np.array([np.array(frame["transform_matrix"])[:3,3]\
                for frame in [frames for frames in transforms["frames"]]])

        min_x = np.min(P[:,0])
        min_y = np.min(P[:,1])
        min_z = np.min(P[:,2])

        max_x = np.max(P[:,0])
        max_y = np.max(P[:,1])
        max_z = np.max(P[:,2])

        box_len = [max_x - min_x, max_y - min_y, max_z - min_z]
        box_center = np.array([max_x + min_x, max_y + min_y, max_z + min_z])/2

        cube_len = [0, max(box_len)]
        cube = np.array(list(product(cube_len,cube_len,cube_len)))
        unit_cube_center = np.array([cube_len[1]]*3)/2
        minimal_cube = cube + (box_center - unit_cube_center)

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(P[:,0], P[:,1], P[:,2])
        ax.scatter(minimal_cube[:,0],minimal_cube[:,1], minimal_cube[:,2], color='r')
        print(f"The maximal required length of shot rays are: {max(box_len)}, make sure that created rays atleast exceedes this size")
        plt.show()




