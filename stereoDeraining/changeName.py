import cv2

for i in range(1800):
    img_gt = cv2.imread('rain100H/gt/norain-%d.png' %(i+1))
    img_rain = cv2.imread('rain100H/rain/norain-%dx2.png' %(i+1))
    cv2.imwrite('Rain100H/gt/%d.png' %(i+1), img_gt)
    cv2.imwrite('Rain100H/rain/%d.png' % (i+1), img_rain)