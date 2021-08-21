import cv2
import argparse
import numpy as np

import imgFuncs as utils
from kmeans import kMeans
from ditherFuncs import serpentineKTone

def kColor(path, k, sample_factor, lnorm, output_path, hard_coded):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #print(img.shape) 

    img = utils.toDouble(img)
    downSampled = utils.downSample(img, sample_factor)
    means = kMeans(downSampled, k, lnorm, hard_coded)
    print(f"Using colors {means}")

    #biTone = serpentineKTone(toDouble(img), np.array([[1,1,1],[0,0,0]])) 
    biTone = serpentineKTone(downSampled, means)
    final = utils.toByte(utils.upSample(biTone, sample_factor))
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
    parser.add_argument('--static_color','-sc', action='append', default=[], type=str,
                        help='Add a static color in RGB hex format that will not be updated during k clustering. Usage -sc 000000 -sc FFFFFF to add black and white.')
    
    parser.add_argument('path', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if(len(args.path) != 1):
        print("path to input image positional argument is required")
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

    if(len(args.static_color) > args.clusters):
        print("Can't have more static colors than total colors")
        exit()

    #print(args.static_color)

    color_arr = []
    for colorStr in args.static_color:
        try:
            color_arr.append(hexToRGB(colorStr))
        except Exception:
            print(f"Error parsing hex string for static color {colorStr}")
            exit()
    
    #print(color_arr)
    hard_coded = np.array(color_arr)
    #print(hard_coded)


    kColor(args.path[0], args.clusters, args.sample_size, args.norm, args.output_file, hard_coded)

