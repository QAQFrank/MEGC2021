import os
import cv2

iml = cv2.imread('./testl.jpg')
imr = cv2.imread('./testr1.jpg')
print(iml.shape)
for i in range(64, 127):
    iml[:, i, :] = imr[:, i, :]

cv2.imwrite('testlrlrlr.png', iml)
