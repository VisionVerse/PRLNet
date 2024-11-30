# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2
from PIL import Image


def direct_field(a, norm=True):
    if a.ndim == 3:
        a = np.squeeze(a)

    h, w = a.shape

    a_Image = Image.fromarray(a)
    a = a_Image.resize((w, h), Image.NEAREST)
    a = np.array(a)

    accumulation = np.zeros((2, h, w), dtype=np.float32)

    # RGB-T数据新增
    if a.max() > 200:
        a = (a/a.max()).astype(np.uint8)
    else:
        a = a.astype(np.uint8)

    for i in np.unique(a)[1:]:
        # b, ind = ndimage.distance_transform_edt(a==i, return_indices=True)
        # c = np.indices((h, w))
        # diff = c - ind
        # dr = np.sqrt(np.sum(diff ** 2, axis=0))

        img = (a == i).astype(np.uint8)
        # cv2.distanceTransform()
        dst, labels = cv2.distanceTransformWithLabels(
            img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
        )

        index = np.copy(labels)
        index[img > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]

        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y

        grid = np.indices(img.shape)
        grid = grid.astype(float)  # <class 'tuple'>: (2, 480, 640)

        diff = grid - nearPixel
        if norm:
            dr = np.sqrt(np.sum(diff**2, axis=0))
        else:
            dr = np.ones_like(img)

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, img > 0] = np.divide(diff[0, img > 0], dr[img > 0])
        direction[1, img > 0] = np.divide(diff[1, img > 0], dr[img > 0])

        accumulation[:, img > 0] = 0
        accumulation = accumulation + direction

    return accumulation