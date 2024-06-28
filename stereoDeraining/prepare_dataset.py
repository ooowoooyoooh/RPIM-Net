import os
import cv2
import argparse
import sys


sys.path.append("..")

def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='haze', help='The input image path')
    parser.add_argument('--output_image_path', type=str, default='datasets/train', help='The output image path')

    return parser.parse_args()


def prepare_dataset(input_image_path, output_image_path):

    for filename in os.listdir(input_image_path):
        (image_name, extension) = os.path.splitext(filename)
        image = cv2.imread(input_image_path + '/' + filename, cv2.IMREAD_COLOR)
        cv2.imwrite(output_image_path + '/' + 'haze-' + image_name + '.png', image)

    return

if __name__ == '__main__':
    #init args
    args = init_args()

    #prepare_dataset
    prepare_dataset(args.input_image_path, args.output_image_path)
