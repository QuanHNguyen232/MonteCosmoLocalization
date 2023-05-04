import cv2
import numpy as np
import os

#Takes an image and crops top and bottom by arbitrary value, then returns.
def crop_img(img):
    img_path = os.path.realpath(img) #get path of original img file
    orig_name = os.path.basename(img_path) #get name of orig img file
    
    crop_img = get_img(img)
    upperHB =  np.size(crop_img, 0)-40 #upper height bound
    upperWB = np.size(crop_img, 1)         #upper width bound
    cropped_img = crop_img[40:upperHB, 0:upperWB] # Slicing to crop the image, first range is height, second is width    
    save_img(cropped_img, './cozmo-images-kidnap/c-' + orig_name)
    return cropped_img

#to stitch images together in a panaorama & crop it 
def stitching():
    images = []
    for i in range(29): #our directory of images has 30 to stitch together
        images.append( #replace directory with your own 
            cv2.imread(f'./cozmo-images-kidnap/{i}-{i*12.0}.jpg'))
    stitcher = cv2.Stitcher.create()
    ret, pano = stitcher.stitch(images)
    #print(pano.shape)
    save_img(pano, './cozmo-images-kidnap/Panorama.jpg')
    #cv2.imwrite('Panorama.jpg', pano)
    #crop image
    crop_img('./cozmo-images-kidnap/Panorama.jpg')

def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey()   # press any key to close
    cv2.destroyAllWindows()

def get_img(filename):
    cv2_img = cv2.imread(filename, 0)    # gray scale --> 1 channel
    return np.array(cv2_img)

def save_img(img: np.ndarray, filepath: str):
    cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

def normalize_img(img):
    scale = 255.0 if img.max() > 200.0 else img.max()
    return np.array(img)/scale

def get_sobel(img, type_sobel='scharr'):
    ksize = -1 if type_sobel=='scharr' else 3
    cv2_sobel_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    cv2_sobel_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    cv2_sobel_x, cv2_sobel_y = cv2.convertScaleAbs(cv2_sobel_x), cv2.convertScaleAbs(cv2_sobel_y)
    cv2_sobel = cv2.addWeighted(cv2_sobel_x, 0.5, cv2_sobel_y, 0.5, 0)
    return cv2_sobel

def convolution(img, kernel, stride=1, pad=0):
    i_h, i_w = img.shape
    k_h, k_w = kernel.shape
    out_size = (int((i_h - k_h + 2*pad)/stride + 1), int((i_w - k_w + 2*pad)/stride + 1))
    out_img = np.zeros(out_size)
    if pad > 0:
        img_padd = np.zeros((i_h + pad*2, i_w + pad*2))
        img_padd[int(pad) : int(-1*pad) , int(pad) : int(-1*pad)] = img
    else: img_padd = img
    for r in range(0, i_h-k_h + 1, stride):
        for c in range(0, i_w-k_w + 1, stride):
            sub_img = img_padd[r:r+k_h, c:c+k_w]
            assert sub_img.shape == kernel.shape, 'error in getting sub_img, size does not match kernel'
            out_img[int(r/stride), int(c/stride)] = np.multiply(sub_img, kernel).sum()
    return out_img

def pooling(img, pool_size=2, stride=2, pad=0, pool_type='max'):
    i_h, i_w = img.shape
    out_size = (int((i_h - pool_size + 2*pad)/stride + 1), int((i_w - pool_size + 2*pad)/stride + 1))
    out_img = np.zeros(out_size)
    if pad > 0:
        img_padd = np.zeros((i_h + pad*2, i_w + pad*2))
        img_padd[int(pad) : int(-1*pad) , int(pad) : int(-1*pad)] = img
    else: img_padd = img
    for r in range(0, i_h-pool_size + 1, stride):
        for c in range(0, i_w-pool_size + 1, stride):
            sub_img = img_padd[r:r+pool_size, c:c+pool_size]
            if pool_type == 'min':
                out_img[int(r/stride), int(c/stride)] = sub_img.min()
            elif pool_type == 'average':
                out_img[int(r/stride), int(c/stride)] = sub_img.mean()
            else:
                out_img[int(r/stride), int(c/stride)] = sub_img.max()
    return out_img

def get_kernel(kerel_type):
    if kerel_type == 'box_blur':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    elif kerel_type == 'canny_edge_detect':
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    elif kerel_type == 'canny_edge_detect_2':
        return np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    elif kerel_type == 'gauss_blur':
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    elif kerel_type == 'prewitt_vert':
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    elif kerel_type == 'prewitt_horiz':
        return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif kerel_type == 'sobel_vert':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif kerel_type == 'sobel_horiz':
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif kerel_type == 'laplacian':
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif kerel_type == 'emboss':
        return np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    elif kerel_type == 'sharpen':
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

#if __name__ == '__main__':
    #DIR = 'cozmo-images-kidnap - Copy'

    # idx = 1
    # imgname = f'{DIR}/{idx}-{idx * 18.0}.jpg'
    # cv2_img = get_img(imgname)
    # print(cv2_img.shape)
    # show_img(normalize_img(cv2_img))
    # print(cv2_img.max())

    # cv2_sobel = get_sobel(cv2_img, 'scharr')
    # show_img(cv2_sobel)

    # output = convolution(normalize_img(cv2_img), get_kernel('x')*2, stride=1, pad=1)
    # print(output.shape)
    # show_img(output)
    # print(output.max())

    # output = convolution(normalize_img(output), get_kernel('canny_edge_detect'), stride=1, pad=1)
    # print(output.shape)
    # show_img(output)
    # print(output.max())

    

    # output1 = pooling(output, pool_size=2, stride=2, pad=0, pool_type='max')
    # print(output1.shape)
    # show_img(output1)
    
    #test image stiching
    # img = get_img('./cozmo-images-kidnap/Panorama.jpeg')
    # show_img(img)
    
    #test cropping
    #img = crop_img('cozmo-images-kidnap - Copy\kidnapPhoto.jpg')
    #show_img(img)
    
    #stitching()
    #print("Done stitching")

    