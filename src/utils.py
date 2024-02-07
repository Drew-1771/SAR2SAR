import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.ndimage
from scipy import special
from pathlib import Path


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


def normalize_sar(im):
    return ((np.log(np.clip(im, 0.24, np.max(im))) - m) * 255 / (M - m)).astype(
        "float32"
    )


def denormalize_sar(im):
    return np.exp((M - m) * (np.squeeze(im)).astype("float32") + m)


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
    im = Image.fromarray(im.astype("float64")).convert("L")
    # check to see if file exists
    filename = filename.replace("npy", "png")
    if not Path(filename).exists():
        Path(filename).touch(exist_ok=True)
    im.save(filename)


def save_sar_images(
    denoised,
    noisy,
    imagename,
    save_dir,
    store_noisy: bool,
    generate_png: bool,
    debug: bool,
):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir()
    if debug:
        print(f"[*] Saving images to {save_dir}")

    threshold = np.mean(noisy) + 3 * np.std(noisy)
    if threshold == 0:
        generate_png = False
        print("\t[!] Threshold calculated to be 0, could not store as PNG properly")

    denoisedfilename = Path(save_dir) / str("denoised_" + imagename)
    if not denoisedfilename.exists():
        denoisedfilename.touch(exist_ok=True)
    denoisedfilename = str(denoisedfilename)
    np.save(denoisedfilename, denoised)
    if debug:
        print(f"\t[*] Saved to {denoisedfilename}")
    if generate_png:
        store_data_and_plot(denoised, threshold, denoisedfilename)
        if debug:
            print(f"\t[*] Saved png of {denoisedfilename.replace('npy', 'png')}")

    if store_noisy:
        noisyfilename = Path(save_dir) / str("noisy_" + imagename)
        if not noisyfilename.exists():
            noisyfilename.touch(exist_ok=True)
        noisyfilename = str(noisyfilename)
        np.save(noisyfilename, noisy)
        if debug:
            print(f"\t[*] Saved to {noisyfilename}")
        if generate_png:
            store_data_and_plot(noisy, threshold, noisyfilename)
            if debug:
                print(f"\t[*] Saved png of {noisyfilename.replace('npy', 'png')}")
