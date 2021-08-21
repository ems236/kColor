import numpy as np
from datetime import datetime

import convFuncs as conv
import imgFuncs as utils

def kTonePx(img, x, y, loss, tones, lossConv):
    start = datetime.now()

    _,color,error,_ = utils.closest(img[y,x] + loss[y,x], tones)
    closestTime = datetime.now()
    #print(f"closest took {closestTime - start}")
    img[y,x] = color
    loss[y,x] = error
    loss = conv.applyConvToPx(loss, x, y, lossConv)
    #print(f"conv took {datetime.now() - closestTime}")

    return loss

def serpentineKTone(img, tones):
    rightLoss = np.array([[0,0,0],[0,0,7/16],[3/16,5/16,1/16]])
    leftLoss = np.array([[0,0,0],[7/16,0,0],[1/16,5/16,3/16]])

    rows,cols,_ = img.shape
    loss = utils.emptyImage(rows, cols)
    for y in range(rows):
        if y % 2 == 0:
            for x in range(cols):
                loss = kTonePx(img, x, y, loss, tones, rightLoss)
        else:
            for x in range(cols - 1, -1, -1):
                loss = kTonePx(img, x, y, loss, tones, leftLoss)

    return img

