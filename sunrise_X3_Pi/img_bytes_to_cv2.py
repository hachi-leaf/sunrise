import numpy as np
import cv2

# 将mipi摄像头获取的bytes图像转化为cv2的BGR图像
def img_bytes_to_cv2(img_byte,w,h):
    """
    The image data that mipi camera read is class bytes,
    So this function can change bytes image into cv2 image.
    """
    nv12_img = np.frombuffer(img_byte,dtype=np.uint8).reshape((int(h*1.5),w))
    bgr_img = cv2.cvtColor(nv12_img,cv2.COLOR_YUV420SP2RGB)

    return bgr_img
