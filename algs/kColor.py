import cv2
import argparse
import numpy as np
import re

import kColor.algs.imgFuncs as utils
from kColor.algs.kmeans import kMeans
from kColor.algs.ditherFuncs import serpentineKTone

def kColor(inImg, k, sample_factor, lnorm, hard_coded):
    print(f"""Running with args:
    k:{k}
    sample_factor:{sample_factor}
    lnorm:{lnorm}
    hard_coded:{hard_coded}
    dimensions:{inImg.shape}
    """)

    img = utils.toDouble(inImg)
    downSampled = utils.downSample(img, sample_factor)
    means = kMeans(downSampled, k, lnorm, hard_coded)
    print(f"Using colors {means}")

    #biTone = serpentineKTone(toDouble(img), np.array([[1,1,1],[0,0,0]])) 
    biTone = serpentineKTone(downSampled, means)
    final = utils.toByte(utils.upSample(biTone, sample_factor))
    #cv2.imshow("got those peps", final)
    #cv2.imshow("got those og peps", cv2.imread("peps.jpeg", cv2.IMREAD_COLOR))

    return final
    #cv2.imwrite(output_path, final)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def hexListToRgb(colors):
    color_arr = []
    for colorStr in colors:
        color_arr.append(hexToRGB(colorStr))
    return color_arr

def kColorFromDict(imgbytes, someDict):
    print("got to the dict")
    outName = someDict["outName"][0]
    k = int(someDict["clusters"][0])
    colors = []
    if "colors" in someDict:
        colors = hexListToRgb(someDict["colors"])
    lnorm = int(someDict["lnorm"][0])
    sample_factor = int(someDict["sample_size"][0])

    print("read the vals")

    if k <= 0:
        raise ValueError("k must be positive")
    if len(colors) > k:
        raise ValueError("can't have more static colors than clusters")
    if lnorm <= 0:
        raise ValueError("must use a positive norm")
    if sample_factor <= 0:
        raise ValueError("must use a positive sampling rate")
    if len(outName) <= 0 or outName.rfind(".") < 0:
        raise ValueError("must have an output file name")
    
    print("checked the vals")

    ext = outName[outName.rfind(".") :]

    print("ext is " + ext)

    print(type(imgbytes))
    nparr = np.frombuffer(imgbytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    newBytes = kColor(img, k, sample_factor, lnorm, colors)
    print("ran the alg")

    success, bytearr = cv2.imencode(ext, newBytes)
    print("wrote to bytes")

    if success:
        return bytearr, outName
    
    raise ValueError("unable to write file")
    

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
    try:
        color_arr = hexListToRgb(args.static_color)
    except Exception:
            print(f"Error parsing hex string for static color")
            exit()

    #print(color_arr)
    hard_coded = np.array(color_arr)
    #print(hard_coded)

    img = cv2.imread(args.path[0], cv2.IMREAD_COLOR)
    outBytes = kColor(img, args.clusters, args.sample_size, args.norm, hard_coded)
    cv2.imwrite(args.output_file, outBytes)

