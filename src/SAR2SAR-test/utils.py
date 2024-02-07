import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.ndimage
from scipy import special


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

def normalize_sar(im):
    return ((np.log(np.clip(im,0.24,np.max(im))) - m) * 255 / (M - m)).astype('float32')

def denormalize_sar(im):
    return np.exp((M - m) * np.clip((np.squeeze(im)).astype('float32'),0,1) + m)

def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        im = normalize_sar(im)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        im = normalize_sar(im)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def save_sar_images(denoised, noisy, imagename, save_dir):
    choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92,
        'lely':235.90, 'ramb':167.22, 'risoul':306.94, 'limagne':178.43}
    threshold = np.mean(noisy)+3*np.std(noisy)
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = save_dir + "/noisy_" + imagename
    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)
