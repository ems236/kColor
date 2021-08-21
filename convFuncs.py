import numpy as np

def getImgCoord(centerCoord, kernelDim, kernelCoord):
    #assume is odd
    kernelCenter = (kernelDim // 2)
    distFromCenter = kernelCoord - kernelCenter 
    return centerCoord + distFromCenter  

def canConvPixel(rows, cols, imgX, imgY, kernalVal):
    xGood = 0 <= imgX < cols
    yGood = 0 <= imgY < rows
    kernelMeaningful = kernalVal != 0 
    return xGood and yGood and kernelMeaningful

def applyConvToPx(img, x, y, kernel, destImg = None):
    rows,cols,_ = img.shape
    if destImg is None:
        destImg = img
    
    kernel_rows, kernel_cols = kernel.shape
    for i in range(kernel_cols):
        for j in range(kernel_rows):
            imgX = getImgCoord(x, kernel_cols, i)
            imgY = getImgCoord(y, kernel_rows, j)
            kernel_val = kernel[j, i]
            if canConvPixel(rows, cols, imgX, imgY, kernel_val):
                for channel in range(3):
                    destImg[imgY, imgX, channel] = img[imgY, imgX, channel] + kernel_val * img[y,x,channel]
    
    return destImg


def conv(img, kernel):
    #assert(len(kernel.shape()) == 2)
    #assert(kernel.shape()[0] % 2 == 1)
    #assert(kernel.shape()[1] % 2 == 1)
    rows,cols,_ = img.shape
    for x in range(cols):
        for y in range(rows):
            #TODO this is wrong because it deals with copies wrong
            applyConvToPx(img, x, y, kernel)
