import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(14):
            input_file = "%d_%d.jpg" % (i+1, j+1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_heavy(input_data_path, target_data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(input_data_path)
    target_path = os.path.join(target_data_path)

    save_target_path = os.path.join(target_data_path, 'train_target.h5')
    save_input_path = os.path.join(input_data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path, target_file)):

            target = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "norain-%dx2.png" % (i + 1)

            if os.path.exists(os.path.join(input_path, input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)
"""

def prepare_data_RainTrainL(input_data_path, target_data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(input_data_path)
    target_path = os.path.join(target_data_path)

    save_target_path = os.path.join(target_data_path, 'train_target.h5')
    save_input_path = os.path.join(input_data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path, target_file)):

            target = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "norain-%dx2.png" % (i + 1)

            if os.path.exists(os.path.join(input_path, input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)
"""
def prepare_data_Raindrop(input_data_path, target_data_path, patch_size, stride):
    print('process training data')
    input_path = os.path.join(input_data_path)
    target_path = os.path.join(target_data_path)

    save_target_path = os.path.join(target_data_path, 'train_target.h5')
    save_input_path = os.path.join(input_data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(861):
        target_file = "%d_clean.png" % i
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "%d_rain.png" % i
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s #samples: %d" % (target_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_HazeTrain(data_path, patch_size, stride):
    #process
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(5000):
        target_file = "gt-%d.png" % (i+1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "haze-%d.png" % (i+1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1 :
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s #samples %d" %(input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_ITS(input_data_path, target_data_path, patch_size, stride):
    #process
    print('process training data')
    input_path = os.path.join(input_data_path)
    target_path = os.path.join(target_data_path)

    save_target_path = os.path.join(target_data_path, 'train_target.h5')
    save_input_path = os.path.join(input_data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for inputname in os.listdir(input_path):
        if inputname.endswith('.png'):
            gt_name = inputname.strip()
            targetname = gt_name.split('_')[0]
            target_name = targetname + '.png'
            print(target_name)
            print(inputname)

            input_image = cv2.imread(os.path.join(input_path, inputname))
            target_image = cv2.imread(os.path.join(target_path, target_name))

        #print()
            b, g, r = cv2.split(input_image)
            input_image = cv2.merge([r, g, b])
            b, g, r = cv2.split(target_image)
            target_image = cv2.merge([r, g, b])

            input_image = np.float32(normalize(input_image))
            input_patches = Im2Patch(input_image.transpose(2, 0, 1), win=patch_size, stride=stride)
            target_image = np.float32(normalize(target_image))
            target_patches = Im2Patch(target_image.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s #samples %d" %(target_name, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_kitti2012_train(data_path, patch_size, stride):
    print('process training data')
    img_gtL_path = os.path.join(data_path, 'image_2')
    img_gtR_path = os.path.join(data_path, 'image_3')
    img_rainL_path = os.path.join(data_path, 'image_2_rain50')
    img_rainR_path = os.path.join(data_path, 'image_3_2_rain50')

    save_gtL_path = os.path.join(data_path, 'train_gtL.h5')
    save_gtR_path = os.path.join(data_path, 'train_gtR.h5')
    save_rainL_path = os.path.join(data_path, 'train_rainL.h5')
    save_rainR_path = os.path.join(data_path, 'train_rainR.h5')

    gtL_h5f = h5py.File(save_gtL_path, 'w')
    gtR_h5f = h5py.File(save_gtR_path, 'w')
    rainL_h5f = h5py.File(save_rainL_path, 'w')
    rainR_h5f = h5py.File(save_rainR_path, 'w')
    train_num = 0
    for i in range(194):
        for j in range(21):
            img_gtL = os.path.join(img_gtL_path, '%06d_%02d_norain_2.png' % (i, j))
            img_gtR = os.path.join(img_gtR_path, '%06d_%02d_norain_3.png' % (i, j))
            img_rainL = os.path.join(img_rainL_path, '%06d_%02d_rain_2_50.jpg' % (i, j))
            img_rainR = os.path.join(img_rainR_path,'%06d_%02d_rain_0_3_50.jpg' % (i, j))
            if os.path.exists(img_gtL) and os.path.exists(img_gtR) and os.path.exists(img_rainL) and os.path.exists(img_rainR):
                gtL = cv2.imread(img_gtL)
                b, g, r = cv2.split(gtL)
                gtL = cv2.merge([r, g, b])
                gtR = cv2.imread(img_gtR)
                b, g, r = cv2.split(gtR)
                gtR = cv2.merge([r, g, b])
                rainL = cv2.imread(img_rainL)
                b, g, r = cv2.split(rainL)
                rainL = cv2.merge([r, g, b])
                rainR = cv2.imread(img_rainR)
                b, g, r = cv2.split(rainR)
                rainR = cv2.merge([r, g, b])
                gtL = np.float32(normalize(gtL))
                gtL_patches = Im2Patch(gtL.transpose(2,0,1), win=patch_size, stride=stride)
                gtR = np.float32(normalize(gtR))
                gtR_patches = Im2Patch(gtR.transpose(2,0,1), win=patch_size, stride=stride)
                rainL = np.float32(normalize(rainL))
                rainL_patches = Im2Patch(rainL.transpose(2,0,1), win=patch_size, stride=stride)
                rainR = np.float32(normalize(rainR))
                rainR_patches = Im2Patch(rainR.transpose(2,0,1), win=patch_size, stride=stride)

                print("rain file: %s # samples: %d" % ('%06d_%02d_rain_2_50.jpg' % (i, j), rainL_patches.shape[3]))
                for n in range(rainL_patches.shape[3]):
                    gtL_data = gtL_patches[:, :, :, n].copy()
                    gtL_h5f.create_dataset(str(train_num), data=gtL_data)
                    gtR_data = gtR_patches[:, :, :, n].copy()
                    gtR_h5f.create_dataset(str(train_num),data=gtR_data)
                    rainL_data = rainL_patches[:, :, :, n].copy()
                    rainL_h5f.create_dataset(str(train_num), data=rainL_data)
                    rainR_data = rainR_patches[:, :, :, n].copy()
                    rainR_h5f.create_dataset(str(train_num), data=rainR_data)

                    train_num+=1

    gtL_h5f.close()
    gtR_h5f.close()
    rainL_h5f.close()
    rainR_h5f.close()
    print("training set, #samples %d\n" % train_num)

def prepare_data_kitti2012_single_train(data_path, patch_size, stride):
    print('process training data')
    img_gtL_path = os.path.join(data_path, 'image_2')
    img_gtR_path = os.path.join(data_path, 'image_3')
    img_rainL_path = os.path.join(data_path, 'image_2_rain50')
    img_rainR_path = os.path.join(data_path, 'image_3_2_rain50')

    save_gt_path = os.path.join(data_path, 'train_gt.h5')
    # save_gtR_path = os.path.join(data_path, 'train_gtR.h5')
    save_rain_path = os.path.join(data_path, 'train_rain.h5')
    # save_rainR_path = os.path.join(data_path, 'train_rainR.h5')

    gt_h5f = h5py.File(save_gt_path, 'w')
    # gtR_h5f = h5py.File(save_gtR_path, 'w')
    rain_h5f = h5py.File(save_rain_path, 'w')
    # rainR_h5f = h5py.File(save_rainR_path, 'w')
    train_num = 0
    for i in range(194):
        for j in range(21):
            img_gtL = os.path.join(img_gtL_path, '%06d_%02d_norain_2.png' % (i, j))
            img_gtR = os.path.join(img_gtR_path, '%06d_%02d_norain_3.png' % (i, j))
            img_rainL = os.path.join(img_rainL_path, '%06d_%02d_rain_2_50.jpg' % (i, j))
            img_rainR = os.path.join(img_rainR_path,'%06d_%02d_rain_0_3_50.jpg' % (i, j))
            if os.path.exists(img_gtL) and os.path.exists(img_gtR) and os.path.exists(img_rainL) and os.path.exists(img_rainR):
                gtL = cv2.imread(img_gtL)
                b, g, r = cv2.split(gtL)
                gtL = cv2.merge([r, g, b])
                gtR = cv2.imread(img_gtR)
                b, g, r = cv2.split(gtR)
                gtR = cv2.merge([r, g, b])
                rainL = cv2.imread(img_rainL)
                b, g, r = cv2.split(rainL)
                rainL = cv2.merge([r, g, b])
                rainR = cv2.imread(img_rainR)
                b, g, r = cv2.split(rainR)
                rainR = cv2.merge([r, g, b])
                gtL = np.float32(normalize(gtL))
                gtL_patches = Im2Patch(gtL.transpose(2,0,1), win=patch_size, stride=stride)
                gtR = np.float32(normalize(gtR))
                gtR_patches = Im2Patch(gtR.transpose(2,0,1), win=patch_size, stride=stride)
                rainL = np.float32(normalize(rainL))
                rainL_patches = Im2Patch(rainL.transpose(2,0,1), win=patch_size, stride=stride)
                rainR = np.float32(normalize(rainR))
                rainR_patches = Im2Patch(rainR.transpose(2,0,1), win=patch_size, stride=stride)

                print("rain file: %s # samples: %d" % ('%06d_%02d_rain_2_50.jpg' % (i, j), rainL_patches.shape[3]))
                for n in range(rainL_patches.shape[3]):
                    gtL_data = gtL_patches[:, :, :, n].copy()
                    gt_h5f.create_dataset(str(train_num), data=gtL_data)
                    rainL_data = rainL_patches[:, :, :, n].copy()
                    rain_h5f.create_dataset(str(train_num), data=rainL_data)

                    train_num+=1
                    gtR_data = gtR_patches[:, :, :, n].copy()
                    gt_h5f.create_dataset(str(train_num),data=gtR_data)
                    rainR_data = rainR_patches[:, :, :, n].copy()
                    rain_h5f.create_dataset(str(train_num), data=rainR_data)

                    train_num+=1

    gt_h5f.close()
    rain_h5f.close()
    print("training set, #samples %d\n" % train_num)

def prepare_data_kitti15_train(data_path, patch_size, stride):
    print('process training data')
    img_gtL_path = os.path.join(data_path, 'image_2')
    img_gtR_path = os.path.join(data_path, 'image_3')
    img_rainL_path = os.path.join(data_path, 'image_2_rain50')
    img_rainR_path = os.path.join(data_path, 'image_3_rain50')
    save_gtL_path = os.path.join(data_path, 'train_gtL.h5')
    save_gtR_path = os.path.join(data_path, 'train_gtR.h5')
    save_rainL_path = os.path.join(data_path, 'train_rainL.h5')
    save_rainR_path = os.path.join(data_path, 'train_rainR.h5')

    gtL_h5f = h5py.File(save_gtL_path, 'w')
    gtR_h5f = h5py.File(save_gtR_path, 'w')
    rainL_h5f = h5py.File(save_rainL_path, 'w')
    rainR_h5f = h5py.File(save_rainR_path, 'w')
    train_num = 0
    for i in range(200):
        for j in range(21):
            img_gtL = os.path.join(img_gtL_path, '%06d_%02d_norain_2.png' % (i, j))
            img_gtR = os.path.join(img_gtR_path, '%06d_%02d_norain_3.png' % (i, j))
            img_rainL = os.path.join(img_rainL_path, '%06d_%02d_rain_2_50.jpg' % (i, j))
            img_rainR = os.path.join(img_rainR_path, '%06d_%02d_rain_3_50.jpg' % (i, j))
            if os.path.exists(img_gtL) and os.path.exists(img_gtR) and os.path.exists(img_rainL) and os.path.exists(img_rainR):
                gtL = cv2.imread(img_gtL)
                b, g, r = cv2.split(gtL)
                gtL = cv2.merge([r, g, b])
                gtR = cv2.imread(img_gtR)
                b, g, r = cv2.split(gtR)
                gtR = cv2.merge([r, g, b])
                rainL = cv2.imread(img_rainL)
                b, g, r = cv2.split(rainL)
                rainL = cv2.merge([r, g, b])
                rainR = cv2.imread(img_rainR)
                b, g, r = cv2.split(rainR)
                rainR = cv2.merge([r, g, b])
                gtL = np.float32(normalize(gtL))
                gtL_patches = Im2Patch(gtL.transpose(2,0,1), win=patch_size, stride=stride)
                gtR = np.float32(normalize(gtR))
                gtR_patches = Im2Patch(gtR.transpose(2,0,1), win=patch_size, stride=stride)
                rainL = np.float32(normalize(rainL))
                rainL_patches = Im2Patch(rainL.transpose(2,0,1), win=patch_size, stride=stride)
                rainR = np.float32(normalize(rainR))
                rainR_patches = Im2Patch(rainR.transpose(2,0,1), win=patch_size, stride=stride)

                print("rain file: %s # samples: %d" % ('%06d_%02d_rain_2_50.jpg' % (i, j), rainL_patches.shape[3]))
                for n in range(rainL_patches.shape[3]):
                    gtL_data = gtL_patches[:, :, :, n].copy()
                    gtL_h5f.create_dataset(str(train_num), data=gtL_data)
                    gtR_data = gtR_patches[:, :, :, n].copy()
                    gtR_h5f.create_dataset(str(train_num),data=gtR_data)
                    rainL_data = rainL_patches[:, :, :, n].copy()
                    rainL_h5f.create_dataset(str(train_num), data=rainL_data)
                    rainR_data = rainR_patches[:, :, :, n].copy()
                    rainR_h5f.create_dataset(str(train_num), data=rainR_data)

                    train_num+=1

    gtL_h5f.close()
    gtR_h5f.close()
    rainL_h5f.close()
    rainR_h5f.close()
    print("training set, #samples %d\n" % train_num)


def prepare_data_kitti15_single_train(data_path, patch_size, stride):
    print('process training data')
    img_gtL_path = os.path.join(data_path, 'image_2')
    img_gtR_path = os.path.join(data_path, 'image_3')
    img_rainL_path = os.path.join(data_path, 'image_2_rain50')
    img_rainR_path = os.path.join(data_path, 'image_3_rain50')

    save_gt_path = os.path.join(data_path, 'train_gt.h5')
    # save_gtR_path = os.path.join(data_path, 'train_gtR.h5')
    save_rain_path = os.path.join(data_path, 'train_rain.h5')
    # save_rainR_path = os.path.join(data_path, 'train_rainR.h5')

    gt_h5f = h5py.File(save_gt_path, 'w')
    # gtR_h5f = h5py.File(save_gtR_path, 'w')
    rain_h5f = h5py.File(save_rain_path, 'w')
    # rainR_h5f = h5py.File(save_rainR_path, 'w')
    train_num = 0
    for i in range(200):
        for j in range(21):
            img_gtL = os.path.join(img_gtL_path, '%06d_%02d_norain_2.png' % (i, j))
            img_gtR = os.path.join(img_gtR_path, '%06d_%02d_norain_3.png' % (i, j))
            img_rainL = os.path.join(img_rainL_path, '%06d_%02d_rain_2_50.jpg' % (i, j))
            img_rainR = os.path.join(img_rainR_path,'%06d_%02d_rain_3_50.jpg' % (i, j))
            if os.path.exists(img_gtL) and os.path.exists(img_gtR) and os.path.exists(img_rainL) and os.path.exists(img_rainR):
                gtL = cv2.imread(img_gtL)
                b, g, r = cv2.split(gtL)
                gtL = cv2.merge([r, g, b])
                gtR = cv2.imread(img_gtR)
                b, g, r = cv2.split(gtR)
                gtR = cv2.merge([r, g, b])
                rainL = cv2.imread(img_rainL)
                b, g, r = cv2.split(rainL)
                rainL = cv2.merge([r, g, b])
                rainR = cv2.imread(img_rainR)
                b, g, r = cv2.split(rainR)
                rainR = cv2.merge([r, g, b])
                gtL = np.float32(normalize(gtL))
                gtL_patches = Im2Patch(gtL.transpose(2,0,1), win=patch_size, stride=stride)
                gtR = np.float32(normalize(gtR))
                gtR_patches = Im2Patch(gtR.transpose(2,0,1), win=patch_size, stride=stride)
                rainL = np.float32(normalize(rainL))
                rainL_patches = Im2Patch(rainL.transpose(2,0,1), win=patch_size, stride=stride)
                rainR = np.float32(normalize(rainR))
                rainR_patches = Im2Patch(rainR.transpose(2,0,1), win=patch_size, stride=stride)

                print("rain file: %s # samples: %d" % ('%06d_%02d_rain_2_50.jpg' % (i, j), rainL_patches.shape[3]))
                for n in range(rainL_patches.shape[3]):
                    gtL_data = gtL_patches[:, :, :, n].copy()
                    gt_h5f.create_dataset(str(train_num), data=gtL_data)
                    rainL_data = rainL_patches[:, :, :, n].copy()
                    rain_h5f.create_dataset(str(train_num), data=rainL_data)

                    train_num+=1
                    gtR_data = gtR_patches[:, :, :, n].copy()
                    gt_h5f.create_dataset(str(train_num),data=gtR_data)
                    rainR_data = rainR_patches[:, :, :, n].copy()
                    rain_h5f.create_dataset(str(train_num), data=rainR_data)

                    train_num+=1

    gt_h5f.close()
    rain_h5f.close()
    print("training set, #samples %d\n" % train_num)

class Dataset(udata.Dataset):
    def __init__(self, inputdataL_path='.', inputdataR_path='.', targetdataL_path='.', targetdataR_path='.'):
        super(Dataset, self).__init__()

        self.inputdataL_path = inputdataL_path
        self.inputdataR_path = inputdataR_path
        self.targetdataL_path = targetdataL_path
        self.targetdataR_path = targetdataR_path

        inputL_path = os.path.join(self.inputdataL_path, 'train_rainL.h5')
        inputR_path = os.path.join(self.inputdataR_path, 'train_rainR.h5')
        targetL_path = os.path.join(self.targetdataL_path, 'train_gtL.h5')
        targetR_path = os.path.join(self.targetdataR_path, 'train_gtR.h5')

        inputL_h5f = h5py.File(inputL_path, 'r')
        inputR_h5f = h5py.File(inputR_path, 'r')
        targetL_h5f = h5py.File(targetL_path, 'r')
        targetR_h5f = h5py.File(targetR_path, 'r')

        self.keys = list(targetL_h5f.keys())
        random.shuffle(self.keys)
        inputL_h5f.close()
        inputR_h5f.close()
        targetL_h5f.close()
        targetR_h5f.close()

    def __len__(self):
        return  len(self.keys)

    def __getitem__(self, index):
        inputL_path = os.path.join(self.inputdataL_path, 'train_rainL.h5')
        inputR_path = os.path.join(self.inputdataR_path, 'train_rainR.h5')
        targetL_path = os.path.join(self.targetdataL_path, 'train_gtL.h5')
        targetR_path = os.path.join(self.targetdataR_path, 'train_gtR.h5')

        inputL_h5f = h5py.File(inputL_path, 'r')
        inputR_h5f = h5py.File(inputR_path, 'r')
        targetL_h5f = h5py.File(targetL_path, 'r')
        targetR_h5f = h5py.File(targetR_path, 'r')

        key = self.keys[index]
        inputL = np.array(inputL_h5f[key])
        inputR = np.array(inputR_h5f[key])
        targetL = np.array(targetL_h5f[key])
        targetR = np.array(targetR_h5f[key])

        inputL_h5f.close()
        inputR_h5f.close()
        targetL_h5f.close()
        targetR_h5f.close()

        return torch.Tensor(inputL), torch.Tensor(inputR), torch.Tensor(targetL), torch.Tensor(targetR)


class Dataset_Single(udata.Dataset):
    def __init__(self, inputdataL_path='.', targetdataL_path='.'):
        super(Dataset_Single, self).__init__()

        self.inputdataL_path = inputdataL_path
        self.targetdataL_path = targetdataL_path

        inputL_path = os.path.join(self.inputdataL_path, 'train_rainR.h5')
        targetL_path = os.path.join(self.targetdataL_path, 'train_gtR.h5')
        inputL_h5f = h5py.File(inputL_path, 'r')
        targetL_h5f = h5py.File(targetL_path, 'r')

        self.keys = list(targetL_h5f.keys())
        random.shuffle(self.keys)
        inputL_h5f.close()
        targetL_h5f.close()

    def __len__(self):
        return  len(self.keys)

    def __getitem__(self, index):
        inputL_path = os.path.join(self.inputdataL_path, 'train_rainR.h5')
        targetL_path = os.path.join(self.targetdataL_path, 'train_gtR.h5')

        inputL_h5f = h5py.File(inputL_path, 'r')
        targetL_h5f = h5py.File(targetL_path, 'r')

        key = self.keys[index]
        inputL = np.array(inputL_h5f[key])
        targetL = np.array(targetL_h5f[key])

        inputL_h5f.close()
        targetL_h5f.close()

        return torch.Tensor(inputL), torch.Tensor(targetL)



"""
class Dataset(udata.Dataset):
    def __init__(self, inputdata_path='.', targetdata_path='.'):
        super(Dataset, self).__init__()

        self.inputdata_path = inputdata_path
        self.targetdata_path = targetdata_path

        target_path = os.path.join(self.targetdata_path, 'train_target.h5')
        input_path = os.path.join(self.inputdata_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.targetdata_path, 'train_target.h5')
        input_path = os.path.join(self.inputdata_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)
"""

