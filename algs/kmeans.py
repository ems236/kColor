import numpy as np
from random import randrange

import imgFuncs as utils

def classifyKMean(img, means, lnorm, static_color_count):
    rows,cols,_ = img.shape
    classCounts = [0 for _ in means]
    newMeans = np.zeros((len(means), 3))
    error = 0
    for y in range(rows):
        for x in range(cols):
            pxError,_,_,idx = utils.closest(img[y,x], means, lnorm)
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
