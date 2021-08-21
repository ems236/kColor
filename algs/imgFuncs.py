import numpy as np

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

def hexToRGB(hexString):
    if len(hexString) != 6:
        print("incorrect length, expecting 6")
        raise Exception("incorrect length, expecting 6.")
    
    r = int(hexString[0:2], 16)
    g = int(hexString[2:4], 16)
    b = int(hexString[4:6], 16)

    return [r / 255, g / 255, b / 255]


def emptyImage(rows, cols):
    return np.zeros((rows,cols,3))

#returns lnorm error, color, l1 error, option index
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
