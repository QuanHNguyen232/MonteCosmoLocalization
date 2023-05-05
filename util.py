import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps

import cozmo
from cozmo.util import degrees

IMG_DIR = 'cozmo-imgs'

def makeHistogram():
    '''
    Make a histogram to display where cozmo thinks it is (blue - before, orange - after localization)    
    Modified work of: http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/#code
    '''
    print('makeHistogram ...')
    df = pd.read_csv("cozmo-images-kidnap/data.csv")
    pano = cv2.imread("cozmo-images-kidnap/cropped-pano.jpg") # our cropped panorama
    width = pano.shape[1]

    originalPredictions = df['particles']
    newestPredictions = df['newParticles']

    # clf() clears the histogram. We found it compiles at the start, so whatever data is in 
    # the csv file when you run the code will be included in the final histogram with the new
    # data unless you clear it first. 
    plt.clf()
    fig, ax = plt.subplots()
    custom_bins = np.arange(0, width, 10) #create custom bins for histogram -> num predictions per range of 10
    ax.hist(originalPredictions,range = [0,width], bins = custom_bins)
    freq, bins, patches = ax.hist(newestPredictions,range = [0,width], bins = custom_bins) #newest belief predications
    ax.set_title('Cozmo MCL Predictions')
    ax.set_xlabel('Width of Panorama')
    ax.set_ylabel('Frequency of Predicitons')
    fig.savefig("cozmo-images-kidnap/hist.png")
    
    bin_max = np.where(freq == freq.max()) #returns an array, can be multiple maxs that are equal
    max_belief = bin_max[0] #get index where max bin is located in array
    max_belief = max_belief[0] * 10 #scale result up to be pixel location in pano
    
    return max_belief #return max_belief for use by MCL

#Takes an image and crops top and bottom by arbitrary value, then returns.
def crop_img(img: np.ndarray, h_min, h_max, w_min, w_max) -> np.ndarray:
    cropped_img = img[h_min : h_max, w_min : w_max] # Slicing to crop the image, first range is height, second is width    
    return cropped_img

#to stitch images together in a panaorama & crop it 
def stitching():
    img_list = [file for file in os.listdir('./cozmo-images-kidnap/') if (file[0].isdigit() and file.endswith('jpg'))]
    # stitching does not work if include the last img
    images = [get_img_rgb(f'./cozmo-images-kidnap/{i}.jpg') for i in range(len(img_list)-1)]
    stitcher = cv2.Stitcher.create()
    _, pano = stitcher.stitch(images)
    save_img(pano, './cozmo-images-kidnap/pano.jpg')
    h, w = img.shape[:2]
    cropped_pano = crop_img(pano, h_min=40, h_max=h-40, w_min=20, w_max=w-20)
    save_img(cropped_pano, './cozmo-images-kidnap/cropped-pano.jpg')

def show_img(img: np.ndarray) -> None:
    if len(img.shape)==2:   # gray img (h, w)
        img = np.stack([img, img, img], axis=-1)
    pil_img = img_np2PIL(img)
    pil_img.show()

def get_img_gray(filename: str) -> np.ndarray:
    img = cv2.imread(filename, 0)    # gray scale --> 1 channel
    return img

def get_img_rgb(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(img, filepath: str) -> None:
    '''
    img: np.ndarray
    '''
    # img = img_np2PIL(img)
    # img.save(filepath)
    cv2.imwrite(filepath, img)

def normalize_img(img: np.ndarray) -> np.ndarray:
    scale = 255.0 if img.max() > 200.0 else img.max()
    return np.array(img)/scale

def img_np2PIL(img: np.ndarray) -> Image:
    if (img.max()>1):
        pil_img = Image.fromarray(img, 'RGB')
    else:
        pil_img = Image.fromarray(np.uint8(img*255.0), 'RGB')
    return pil_img

if __name__ == '__main__':
    # collect_imgs(num_pic=72, img_dir='./model/data/19')
    img_file = './cozmo-images-kidnap/1.jpg'
    img = get_img_gray(img_file)
    print(img.shape)
    stitching()
    