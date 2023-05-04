import cv2
import numpy as np
import os
from PIL import Image

#Takes an image and crops top and bottom by arbitrary value, then returns.
def crop_img(imgname: str) -> np.ndarray:
    img_path = os.path.realpath(imgname) #get path of original img file
    orig_name = os.path.basename(img_path) #get name of orig img file
    
    crop_img = get_img_gray(imgname)
    upperHB =  np.size(crop_img, 0)-40 #upper height bound
    upperWB = np.size(crop_img, 1)         #upper width bound
    cropped_img = crop_img[40:upperHB, 0:upperWB] # Slicing to crop the image, first range is height, second is width    
    save_img(cropped_img, './cozmo-images-kidnap/c-' + orig_name) #FIX ME WHEN DONE
    return cropped_img

#to stitch images together in a panaorama & crop it 
def stitching():
    images = []
    for i in range(30-1): #our directory of images has 29 to stich togehter
        images.append(cv2.imread(f'./cozmo-images-kidnap/{i}.jpg'))
    stitcher = cv2.Stitcher.create()
    _, pano = stitcher.stitch(images)
    #print(pano.shape)
    # save_img(pano, './cozmo-images-kidnap/Panorama.jpg')
    cv2.imwrite('./cozmo-images-kidnap/Panorama.jpg', pano)
    #crop image
    crop_img('./cozmo-images-kidnap/Panorama.jpg')
    # save_img(cropped_img, './cozmo-images-kidnap/cropped-Panorama.jpg')
    print('stitching DONE')

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
    img: np.ndarray or PIL.Image.Image
    '''
    # img = img_np2PIL(img)
    # img.save(filepath)
    cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

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
    stitching()

    