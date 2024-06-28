import os
import cv2
import numpy as np

def compose_two_image_with_hstack(left, mid, right):
    img1 = cv2.imread(left)
    img2 = cv2.imread(mid)
    img3 = cv2.imread(right)
    image = cv2.imread('dataset/test/kitti12_testing/image_2/000000_00_norain_2.png')
    # print(image)
    image[:,0:500,:] = img1
    image[:, 500:1000, :] = img2
    image[:, 1000::, :] = img3
    cv2.imwrite('results2.jpg', image)

compose_two_image_with_hstack('results/1/000000_00_derainL.png', 'results/2/000000_00_derainL.png', 'results/3/000000_00_derainL.png')