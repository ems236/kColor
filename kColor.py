import cv2
import argparse
import numpy as np
from datetime import datetime
from random import randrange

def emptyImage(rows, cols):
    return np.zeros((rows,cols,3))

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

#returns pair of closest + loss vector
#closest by l2 norm
def closest(val, options):
    best = (float('inf'),0,0,0)
    currentIdx = 0
    for opt in options:
        error = np.subtract(val, opt)
        sumSquare = 0
        for channel in error:
            sumSquare += channel ** 2
        if sumSquare < best[0]:
            best = sumSquare,opt,error,currentIdx
        currentIdx += 1
    return best

def kTonePx(img, x, y, loss, tones, lossConv):
    start = datetime.now()

    _,color,error,_ = closest(img[y,x] + loss[y,x], tones)
    closestTime = datetime.now()
    #print(f"closest took {closestTime - start}")
    img[y,x] = color
    loss[y,x] = error
    loss = applyConvToPx(loss, x, y, lossConv)
    #print(f"conv took {datetime.now() - closestTime}")

    return loss

def serpentineKTone(img, tones):
    rightLoss = np.array([[0,0,0],[0,0,7/16],[3/16,5/16,1/16]])
    leftLoss = np.array([[0,0,0],[7/16,0,0],[1/16,5/16,3/16]])

    rows,cols,_ = img.shape
    loss = emptyImage(rows, cols)
    for y in range(rows):
        if y % 2 == 0:
            for x in range(cols):
                loss = kTonePx(img, x, y, loss, tones, rightLoss)
        else:
            for x in range(cols - 1, -1, -1):
                loss = kTonePx(img, x, y, loss, tones, leftLoss)

    return img

def toDouble(img):
    rows,cols,_ = img.shape
    copy = emptyImage(rows, cols)
    for y in range(rows):
        for x in range(cols):
            copy[y,x] = img[y,x] / 255
    
    return copy

def toByte(img):
    rows,cols,_ = img.shape
    copy = emptyImage(rows, cols)
    for y in range(rows):
        for x in range(cols):
            copy[y,x] = img[y,x] * 255
    
    return copy.astype(np.uint8)

def classifyKMean(img, means):
    rows,cols,_ = img.shape
    classCounts = [0 for _ in means]
    newMeans = np.zeros((len(means), 3))
    error = 0
    for y in range(rows):
        for x in range(cols):
            pxError,_,_,idx = closest(img[y,x], means)
            classCounts[idx] += 1
            newMeans[idx] = np.add(newMeans[idx], img[y,x])
            error += pxError

    for i in range(len(means)):
        if classCounts[i] != 0:
            newMeans[i] /= classCounts[i]
        else:
            newMeans[i] = means[i]

    return error, newMeans

def kMeans(img, k):
    #pick k random starts
    means = np.zeros((k, 3))
    for i in range(k):
        rows,cols,_ = img.shape
        means[i] = img[randrange(rows),randrange(cols)]

    prevError = float('inf')
    error,means = classifyKMean(img, means)

    while error < prevError:
        prevError = error
        error,means = classifyKMean(img, means)

    return means

def downSample(img, factor):
    rows,cols,_ = img.shape
    newRows = rows // factor
    newCols = cols // factor
    downSampled = np.zeros((newRows,newCols,3))
    for y in range(newRows):
        for x in range(newCols):
            startY = y * factor
            startX = x * factor
            mean = np.array([0,0,0])
            for i in range(factor):
                for j in range(factor):
                    mean = np.add(mean, img[startY + j, startX + i])
            
            mean /= factor ** 2
            downSampled[y,x] = mean
    
    return downSampled

def upSample(img, factor):
    rows,cols,_ = img.shape
    newRows = rows * factor
    newCols = cols * factor
    upSampled = np.zeros((newRows,newCols,3))
    for y in range(rows):
        for x in range(cols):
            startY = y * factor
            startX = x * factor
            for i in range(factor):
                for j in range(factor):
                    upSampled[startY + j, startX + i] = img[y,x]
    
    return upSampled

img = cv2.imread("bloom.jpeg", cv2.IMREAD_COLOR)
print(img.shape) 
# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
#cv2.imshow("got those peps", img)

#img = toDouble(img[:200,:200])
img = toDouble(img)
sample_factor = 8
downSampled = downSample(img, sample_factor)
means = kMeans(downSampled, 16)
print(means)

#biTone = serpentineKTone(toDouble(img), np.array([[1,1,1],[0,0,0]])) 
biTone = serpentineKTone(downSampled, means)
final = toByte(upSample(biTone, sample_factor))
cv2.imshow("got those peps", final)
#cv2.imshow("got those og peps", cv2.imread("peps.jpeg", cv2.IMREAD_COLOR))

cv2.imwrite("testKtoned.png", final)
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()