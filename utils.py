import os
from multiprocessing import Pool
import cv2
import numpy as np
from PIL import Image

def process_image(p):
    arr = cv2.imread(p)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    res_img = Image.fromarray(arr).resize((256, 256), resample=Image.BILINEAR)
    arr = np.array(res_img)
    arr = np.expand_dims(arr, axis=0)
    return arr

def load_images(fractal_path:str):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  
    filenames = []
    for dirname, _, fnames in os.walk(fractal_path):
        for fname in fnames:
            if os.path.splitext(fname)[1].lower() in image_extensions:
                filenames.append(os.path.join(dirname, fname))

    with Pool() as pool:
        results = pool.map(process_image, filenames)

    return np.vstack(results)

def add_patch(img:np.ndarray,frac:np.ndarray,s:int) -> np.ndarray:
    w_img = img.copy()
    max_x = img.shape[1] - s
    max_y = img.shape[0] - s

    x = np.random.randint(0,max_x+1)
    y = np.random.randint(0,max_y+1)

    w_img[y:y+s,x:x+s] = w_img[y:y+s,x:x+s] + frac[y:y+s,x:x+s]

    return w_img

def mult_patch(img:np.ndarray,frac:np.ndarray,s:int) -> np.ndarray:
    w_img = img.copy()
    max_x = img.shape[1] - s
    max_y = img.shape[0] - s

    x = np.random.randint(0,max_x+1)
    y = np.random.randint(0,max_y+1)

    w_img[y:y+s,x:x+s] = w_img[y:y+s,x:x+s] * frac[y:y+s,x:x+s]

    return w_img

def rand_pix_mix(img:np.ndarray,frac:np.ndarray,s:int) -> np.ndarray:
    img = img.copy()
    frac = frac.copy()
    mix_img = np.zeros_like(img)
    p = 0.3
    mask = np.random.choice([0, 1], size=(256,256), p=[p, 1-p])
    
    for c in range(img.shape[2]):
        mix_img[:,:,c] = (mask*img[:,:,c])+ ((1-mask)*frac[:,:,c])
    
    return mix_img

def rand_ele_mix(img:np.ndarray,frac:np.ndarray,s:int) -> np.ndarray:
    img = img.copy()
    frac = frac.copy()
    mix_img = np.zeros_like(img)
    for c in range(img.shape[2]):
        p = 0.3
        mask = np.random.choice([0, 1], size=(256,256), p=[p, 1-p])
        mix_img[:,:,c] = (mask*img[:,:,c])+ ((1-mask)*frac[:,:,c])
    return mix_img