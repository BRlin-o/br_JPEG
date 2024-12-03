import cv2
import numpy as np

def FDCT(img, block_size=8, color_space="YUV"):
    _img = np.copy(img)
    fdct = np.zeros(_img.shape, dtype=np.float64)
    if color_space == "YUV":
        _img[:, :, 0] = _img[:, :, 0] - 128.

    for i in range(0, _img.shape[0], block_size):
        for j in range(0, _img.shape[1], block_size):
            for k in range(_img.shape[2]):
                fdct[i:i + block_size, j:j + block_size, k] = cv2.dct(_img[i:i + block_size, j:j + block_size, k])
    return fdct

def IDCT(img, block_size=8, color_space="YUV"):
    _img = np.copy(img).astype(np.float32)
    idct_img = np.zeros(img.shape, dtype=np.float32)

    for i in range(0, idct_img.shape[0], block_size):
        for j in range(0, idct_img.shape[1], block_size):
            for k in range(idct_img.shape[2]):
                idct_img[i:i + block_size, j:j + block_size, k] = cv2.idct(_img[i:i + block_size, j:j + block_size, k])
    
    if color_space == "YUV":
        idct_img += 128
    return idct_img
