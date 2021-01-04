# !/usr/bin/env python3
# coding=utf-8
# author=dave.fang@outlook.com
# create=20171225

import numpy as np

def pad_right_down_corner(img, stride, pad_value):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None] #[None,None,None,None]
    pad[0] = 0  # up #upper padding is 0 because window starts from the top
    pad[1] = 0  # left #left padding is also 0 because window starts from the leftmost
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down #now, h % the new calculated value = 0
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right #now, w % the new calculated value = 0

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1)) #when an axis is 0, it np.tile returns empty array. here, pad[0]=0
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1)) 
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1)) #first axis - outermost list, second - the second inner list 
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad
