import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

count = 0
psnr_L_sum = 0
psnr_R_sum = 0
ssim_L_sum = 0
ssim_R_sum = 0
for i in range(195):
    for j in range(21):
        # derain_L_path = os.path.join('results/kitti12_SD_cpu', '%06d_%02d_derainL.png' % (i, j))
        # derain_R_path = os.path.join('results/kitti12_SD_cpu', '%06d_%02d_derainR.png' % (i, j))
        # derain_L_path = os.path.join('../comparedmethod/K12_stereo_result/stereo_result/image_3_2_rain50', '%06d_%02d_rain_1_2_50.jpg' % (i, j))
        # derain_R_path = os.path.join('../comparedmethod/K12_stereo_result/stereo_result/image_3_2_rain50', '%06d_%02d_rain_0_3_50.jpg' % (i, j))
        # gt_L_path = os.path.join('../comparedmethod/K12_stereo_result/stereo_result/image_2_3_rain50', '%06d_%02d_rain_2_50.jpg' % (i, j))
        # gt_R_path = os.path.join('../comparedmethod/K12_stereo_result/stereo_result/image_2_3_rain50', '%06d_%02d_rain_3_50.jpg' % (i, j))
        # gt_L_path = os.path.join('results/kitti12_SD_cpu', '%06d_%02d_derainL.png' % (i, j))
        # gt_R_path = os.path.join('results/kitti12_SD_cpu', '%06d_%02d_derainR.png' % (i, j))
        # gt_L_path = os.path.join('../iPASSR-main/data/test/kitti12_testing/image_2', '%06d_%02d_norain_2.png' % (i, j))
        # gt_R_path = os.path.join('../iPASSR-main/data/test/kitti12_testing/image_3', '%06d_%02d_norain_3.png' % (i, j))
        derain_L_path = os.path.join('results/singleFin/RResASPPN_left_kitti12', '%06d_%02d_derainL.png' % (i, j))
        derain_R_path = os.path.join('results/singleFin/RResASPPN33.40/_left_kitti12', '%06d_%02d_derainR.png' % (i, j))
        gt_L_path = os.path.join('dataset/test/kitti12_testing/image_2', '%06d_%02d_norain_2.png' % (i, j))
        gt_R_path = os.path.join('dataset/test/kitti12_testing/image_3', '%06d_%02d_norain_3.png' % (i, j))

        if os.path.exists(derain_L_path):
            print(i, j)
            derain_L = cv2.imread(derain_L_path)
            derain_R = cv2.imread(derain_R_path)
            gt_L = cv2.imread(gt_L_path)
            gt_R = cv2.imread(gt_R_path)
            psnr_L = psnr(gt_L, derain_L)
            psnr_R = psnr(gt_R, derain_R)
            psnr_L_sum += psnr_L
            psnr_R_sum += psnr_R
            print(psnr_L, psnr_R)
            # print(psnr_L)
            ssim_L = ssim(gt_L, derain_L, multichannel=True)
            ssim_R = ssim(gt_R, derain_R, multichannel=True)
            ssim_L_sum += ssim_L
            ssim_R_sum += ssim_R
            print(ssim_L, ssim_R)
            # print(ssim_L)
            count+=1
            print(count)

print('psnr_L: {}, ssim_L: {}'.format(psnr_L_sum/count, ssim_L_sum/count))
print('psnr_R: {}, ssim_R: {}'.format(psnr_R_sum/count, ssim_R_sum/count))
print('psnr: {}, ssim: {}'.format((psnr_L_sum+psnr_R_sum)/(2*count), (ssim_L_sum+ssim_R_sum)/(2*count)))
# print('psnr: {}, ssim:{}'.format(psnr_L_sum/count, ssim_L_sum/count))
