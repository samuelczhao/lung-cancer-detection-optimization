from __future__ import print_function
import PIL
import numpy as np
import dicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import threading
import queue 
import scipy.misc
import logging
import dataloader

DATA_DIR="data"
CROPPED_DATA_DIR=os.path.join("data", "cropped_mass", "data", "DOI")

def _load_image(loc, use_dicom=False):
    ''' TODO: refactor this method out '''
    dataloader.load_image(loc, use_dicom)

_DS = None
def _get_index_ds(index_filename="mass_case_description_train_set.csv"):
    loc = os.path.join(DATA_DIR, index_filename)
    if not os.path.isfile(loc):
        raise ValueError("Unable to find indexing file")

    global _DS
    if _DS is None:
        _DS = pd.read_csv(loc)
        _DS = _DS.sample(frac=1, random_state=1).reset_index(drop=True)
    return _DS


def _retrieve_all_images(loc, use_dicom=False):
    imgs = []
    if os.path.isfile(loc):
        try:
            imgs += [_load_image(loc, use_dicom=use_dicom)]
        except ValueError:
            print("Unable to load:", loc)
        return imgs

    for d in os.listdir(loc):
        abs_path = os.path.join(loc, d)
        imgs += _retrieve_all_images(abs_path)
    return imgs


def _get_dirname(index, dir_type="cropped", cycle=True):
    ds = _get_index_ds()

    if cycle:
        index = index % ds.shape[0]

    pid = ds['patient_id'][index]

    image_filename = "Mass-Training_{}_{}_{}_{}".format(ds['patient_id'][index],
            ds['side'][index], ds['view'][index], ds['abn_num'][index])

    if dir_type == "dicom":
        image_dir = os.path.join(DATA_DIR, "DOI", image_filename)
    elif dir_type == "cropped":
        image_dir = os.path.join(CROPPED_DATA_DIR, image_filename)
    else:
        raise ValueError("Invalid type passed: {}".format(dir_type))

    return image_dir


def load_image(index):
    return _retrieve_all_images(_get_dirname(index))


def crop(img):
    while img[:,0].sum() == 0:
        img = img[:,1:]
    while img[0,:].sum() == 0:
        img = img[1:,:]
    while img[:,-1].sum() == 0:
        img = img[:,:-1]
    while img[-1,:].sum() == 0:
        img = img[:-1,:]
    return img


def average_image_size():
    num_imgs = 0
    total_w = 0
    total_h = 0
    ws = []
    hs = []
    for i in range(1318):
        imgs = load_image(i)
        num_imgs += len(imgs)
        for img in imgs:
            cropped = crop(img)
            ws += [cropped.shape[0]]
            hs += [cropped.shape[1]]
            total_w += cropped.shape[0]
            total_h += cropped.shape[1]
        print("\rAverage W: {} H: {}".format(
            total_w / num_imgs, total_h / num_imgs), end="")
    print()

    plt.subplot(121)
    plt.hist(ws)
    plt.subplot(122)
    plt.hist(hs)
    plt.show()


def load_image_with_label(index, cycle=True):
    imgs = load_image(index)
    ds = _get_index_ds()
    if cycle:
        index = index % ds.shape[0]
    lbl = ds['pathology'][index]
    return imgs, [lbl] * len(imgs)



def load_images_with_labels(start, end, verbose=False, cycle=True):
    '''
        Gets images with index [start, end)
    '''
    assert end >= start
    ds = _get_index_ds()
    if not cycle:
        assert end <= ds.shape[0]

    images, labels = [], []
    for i in range(start, end):
        temp_img, temp_lbl = load_image_with_label(i, cycle=True)
        images += temp_img
        labels += temp_lbl
        if verbose:
            print("\rLoading {} of {}".format(i - start, end - start), end="")

    if verbose:
        print()

    return (images, labels)


def convert_lbl_to_ind(lbl, use_dict={"MALIGNANT": 1, "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 2}):
    return list(map(lambda l: use_dict[l], lbl))


def convert_ind_to_lbl(ind, use_dict={"MALIGNANT": 1, "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 2}):
    inv_map = {v: k for k, v in use_dict.items()}
    return list(map(lambda i: inv_map[i], ind))


def resave_cropped(index, dest_dir="data/cropped_mass"):
    from PIL import Image
    d = _get_dirname(index)
    imgs = _retrieve_all_images(d)
    num = 0
    for i in imgs:
        c = crop(i)
        # pilimg = Image.fromarray(c)
        os.makedirs(os.path.join(dest_dir, d), exist_ok=True)
        np.save(os.path.join(dest_dir, d, str(num)) + ".npy", c)
        num += 1


def resave_range_cropped(start, end, dest_dir="data/cropped_mass"):
    for i in range(start, end):
        resave_cropped(i, dest_dir=dest_dir)
        print("\rCropped {} of {}".format(i - start, end - start), end="")
    print()


def _data_loading_job(event, data_queue, start=0, end=1e10,
        load_image=lambda i: load_image(i)):
    ind = start
    while not event.is_set():
        data_queue.put(load_image_with_label(ind % end))
        ind += 1


def async_data_loader(size=(226, 226), start=0, end=1e10, batch_size=32):
    data_queue = queue.Queue(maxsize=100)
    event = threading.Event()
    t = threading.Thread(target=_data_loading_job, args=(event, data_queue,
        start, end))
    t.start()

    ind = 0
    images = []
    labels = []
    while True:
        imgs, lbls = data_queue.get()
        imgs = list(map(
            lambda i: 256.0 * (i - i.min()) / (i.max() - i.min()), imgs))
        imgs = list(map(lambda i: scipy.misc.imresize(i, size), imgs))
        images += imgs
        labels += lbls
        if len(images) >= batch_size:
            yield (np.array(images[:batch_size])[:,:,:,None] / 128.0 - 1,
                    np.array(labels[:batch_size]))
            images = images[batch_size:]
            labels = labels[batch_size:]


def clean_data(imgs, lbls, size=(226,226)):
    imgs = list(map(
        lambda i: 256.0 * (i - i.min()) / (i.max() - i.min()), imgs))
    imgs = list(map(lambda i: scipy.misc.imresize(i, size), imgs))
    imgs = np.array(imgs) / 128.0 - 1
    imgs = imgs[:,:,:,None]
    lbls = np.array(convert_lbl_to_ind(lbls))

    return imgs, lbls


if __name__ == '__main__':
    g = async_data_loader()
