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
#closest by lnorm norm, default is l2
def closest(val, options, lnorm = 2):
    best = (float('inf'),0,0,0)
    currentIdx = 0
    for opt in options:
        error = np.subtract(val, opt)
        sumSquare = 0
        for channel in error:
            sumSquare += channel ** lnorm
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

def classifyKMean(img, means, lnorm, static_color_count):
    rows,cols,_ = img.shape
    classCounts = [0 for _ in means]
    newMeans = np.zeros((len(means), 3))
    error = 0
    for y in range(rows):
        for x in range(cols):
            pxError,_,_,idx = closest(img[y,x], means, lnorm)
            classCounts[idx] += 1
            newMeans[idx] = np.add(newMeans[idx], img[y,x])
            error += pxError

    #only do updates for non-static colors
    for i in range(len(means) - static_color_count):
        if classCounts[i] != 0:
            newMeans[i] /= classCounts[i]
        else:
            newMeans[i] = means[i]

    for i in range(len(means) - static_color_count, len(means)):
        newMeans[i] = means[i]

    return error, newMeans

def kMeans(img, k, lnorm, hard_coded):
    #pick k random starts
    means = np.zeros((k, 3))
    static_color_count = len(hard_coded)
    for i in range(k - static_color_count):
        rows,cols,_ = img.shape
        means[i] = img[randrange(rows),randrange(cols)]

    for i in range(static_color_count):
        start = k - static_color_count
        means[i + start] = hard_coded[i]

    prevError = float('inf')
    error,means = classifyKMean(img, means, lnorm, static_color_count)

    EPSILON = 0.01
    while prevError - error > EPSILON:
        prevError = error
        error,means = classifyKMean(img, means, lnorm, static_color_count)

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

def kColor(path, k, sample_factor, lnorm, output_path, hard_coded):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #print(img.shape) 

    img = toDouble(img)
    downSampled = downSample(img, sample_factor)
    means = kMeans(downSampled, k, lnorm, hard_coded)
    print(means)

    #biTone = serpentineKTone(toDouble(img), np.array([[1,1,1],[0,0,0]])) 
    biTone = serpentineKTone(downSampled, means)
    final = toByte(upSample(biTone, sample_factor))
    #cv2.imshow("got those peps", final)
    #cv2.imshow("got those og peps", cv2.imread("peps.jpeg", cv2.IMREAD_COLOR))

    cv2.imwrite(output_path, final)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conver an image into a k-toned downsampled representation. Tones are selected with k-means clustering')
    parser.add_argument('--clusters', "-k", type=int, nargs='?', default=16,
                        help='K number of colors to use. Defaults to 16')
    parser.add_argument('--sample_size', "-s", type=int, nargs='?', default=8,
                        help='downsampling factor to use. If this is n, every nxn pixels in the original image will be 1 pixel in the downsampled image. Default is 8')
    parser.add_argument('--output_file', "-o", type=str, nargs='?', default='myImage.png',
                        help='Output file path to write results to. Default to myImage.png')
    parser.add_argument('--norm', "-l", type=int, nargs='?', default=2,
                        help='Norm to use when calculating distance in k clustering. Default is l2 norm. Infinity norm is not supported.')
    parser.add_argument('path', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if(len(args.path) != 1):
        print("path positional argument is required")
        exit()
    if(args.clusters <= 0):
        print("clusters must be positive")
        exit()
    if(args.sample_size <= 0):
        print("sample size must be positive")
        exit()
    if(args.norm <= 0):
        print("norm must be positive")
        exit()

    hard_coded = np.array([[0,0,0], [1,1,1]])
    #print(hard_coded)


    kColor(args.path[0], args.clusters, args.sample_size, args.norm, args.output_file, hard_coded)

