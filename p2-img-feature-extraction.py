#!/usr/bin/env python2.7

import argparse
import warnings
import cv2
import pysteg.features as features


# --------------------------------------------

# FUNCTIONS

# --------------------------------------------


# FUNCTION: GET FARID FEATURES
def get_farid_features(file_path):

    # get image file from file path
    img = cv2.imread(file_path)

    # copy file x3 for each colour channel
    img_red = img.copy()
    img_green = img.copy()
    img_blue = img.copy()

    # isolate colour channels
    img_red[:, :, 1] = 0
    img_red[:, :, 2] = 0
    img_green[:, :, 0] = 0
    img_green[:, :, 2] = 0
    img_blue[:, :, 0] = 0
    img_blue[:, :, 1] = 0

    # get features for each colour channel
    colour_channels = [img_red, img_green, img_blue]
    for channel in colour_channels:
        print features.farid36(channel)


# MAIN FUNCTION: GLOBAL VARIABLES
if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)  # ignore UserWarnings

    # argument parsing
    parser = argparse.ArgumentParser(description='A helper script for getting image features.')
    parser.add_argument('target_features', action='store', help='Name of target feature group')
    parser.add_argument('file_path', action='store', help='Path for image file')
    args = parser.parse_args()

    # handle arguments
    target = args.target_features
    file_path = args.file_path

    if target == 'farid':
        get_farid_features(file_path)


