"""
    Retrieves data downloaded using the TCIA query tool from NCIA
"""
from __future__ import print_function
from dataloader import *
import dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import hashlib
import random
import shutil
import scipy.misc
import timeout_decorator
import threading

DATA_DIR = "./data/BetterLungCancer/NLST"
CSV_FILE = "./data/BetterLungCancer/label_with_count.csv"
PID_LABEL = 'LungCancerDiagnosis.pid'
TRUE_LABEL = 'Follow-up collected - Confirmed Lung Cancer'
FALSE_LABEL = 'Follow-up Collected - Confirmed Not Lung Cancer'
LUNG_CANCER_LABEL = 'LungCancerDiagnosis.conflc'
FAST_DATA_DIR = "/home/peachball/D/gitlab/siemens/data/BetterLungCancer/CompactNLST"

def convert_dicom_to_npy(in_dir, out_dir):
    if os.path.isfile(in_dir):
        from PIL import Image
        print("Converting {}".format(in_dir))
        npimg = load_image(in_dir, use_dicom=True)
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        np.save(os.path.splitext(out_dir)[0], npimg)
        return
    subdirs = os.listdir(in_dir)
    for sd in subdirs:
        convert_dicom_to_npy(
                os.path.join(in_dir, sd), os.path.join(out_dir, sd))


def dataset_image_dim(in_dir=DATA_DIR):
    ws, hs = [], []
    def load_dims(curdir):
        if os.path.isfile(curdir):
            npimg = load_image(curdir, use_dicom=True)
            w, h = max(npimg.shape), min(npimg.shape)
            ws.append(w)
            hs.append(h)
            if len(ws) % 10000 == 0:
                plt.hexbin(ws, hs, extent=(0,2000,0,2000))
                plt.colorbar()
                plt.show()
            else:
                print("\r{}".format(len(ws)), end="")
        else:
            dirs = os.listdir(curdir)
            for d in dirs:
                load_dims(os.path.join(curdir, d))

    load_dims(in_dir)
    plt.scatter(dims)


def balence_dataset(label_file, out_file):
    ''' THIS METHOD IS USELESS '''
    df = pd.read_csv(label_file)
    positive_count = df[df[LUNG_CANCER_LABEL] == TRUE_LABEL].shape[0]
    negative_count = df.shape[0] - positive_count
    assert(negative_count > positive_count)
    new_df = df[df[LUNG_CANCER_LABEL] == TRUE_LABEL].copy()
    negative_samples = df[df[LUNG_CANCER_LABEL] == FALSE_LABEL]
    new_df = new_df.append(negative_samples.head(positive_count))
    new_df.to_csv(out_file)


def move_all(filepath, dest):
    '''
        Moves all files in subfolder of folder into folder
        NOTE: destination directory must already exist
    '''
    if os.path.isfile(filepath):
        dir_hash = hashlib.md5(filepath.encode()).hexdigest()
        filename = dir_hash + os.path.basename(filepath)
        os.rename(filepath, os.path.join(dest, filename))
        return
    dirs = os.listdir(filepath)
    for d in dirs:
        move_all(os.path.join(filepath, d), dest)


def flatten(filepath):
    dirs = os.listdir(filepath)
    for d in dirs:
        subdir = os.path.join(filepath, d)
        if os.path.isdir(subdir):
            move_all(subdir, filepath)
            shutil.rmtree(subdir)


def flatten_data_dir(data_dir=DATA_DIR, verbose=True):
    '''
        Flattens all downloaded directories
    '''
    dirs = os.listdir(data_dir)
    for d in dirs:
        if verbose:
            print("Flattening %s" % d)
        flatten(os.path.join(data_dir, d))


def listfiles(directory):
    return list(filter(
        lambda f: os.path.isfile(os.path.join(directory, f)),
            os.listdir(directory)))


class DataLoader():
    def __init__(self, cv_split=0.1, csv_file=CSV_FILE, data_dir=DATA_DIR,
            pid_label=PID_LABEL, true_label=TRUE_LABEL,
            false_label=FALSE_LABEL, lung_cancer_label=LUNG_CANCER_LABEL,
            refresh_counts=True):
        self.df = df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.pid_label = pid_label
        self.true_label = true_label
        self.false_label = false_label
        self.lung_cancer_label = lung_cancer_label

        # Get counts for each pid
        rows = df.shape[0]
        counts = np.zeros((rows,))
        if 'imageCount' not in self.df or refresh_counts:
            print("Loading image counts")
            for i in range(rows):
                print("\r{}/{}".format(i, rows), end="")
                pid = df.loc[i, pid_label]
                counts[i] = self._count_images(pid)
            print()
            self.df['imageCount'] = pd.Series(counts, index=df.index)
            self.df.to_csv(csv_file)

        min_class_images = int(round(min(self.get_statistics())))
        self._permuted_true_inds = np.random.permutation(min_class_images)
        self._permuted_false_inds = np.random.permutation(min_class_images)

        # Note that train size is half of actual training set size
        self._train_size = int((1 - cv_split) * min_class_images)
        self._cv_size = int(cv_split * min_class_images)

        # Create array of ones and zeros, to mix up true and false labels
        t_true_false = np.zeros((self._train_size * 2,))
        cv_true_false = np.zeros((self._cv_size * 2,))
        t_true_false[:self._train_size] = 1
        cv_true_false[:self._cv_size] = 1

        self._t_tf = np.random.permutation(t_true_false)
        self._cv_tf = np.random.permutation(cv_true_false)

    def _count_images(self, pid):
        return self._count_files(os.path.join(self.data_dir, str(pid)))

    def _count_files(self, dirname):
        ''' Pure python method of counting all files in directory '''
        if os.path.isfile(dirname):
            return 1
        elif not os.path.isdir(dirname):
            return 0
        else:
            dirs = os.listdir(dirname)
            total_count = 0
            for d in dirs:
                total_count += self._count_files(os.path.join(dirname, d))
            return total_count

    def get_statistics(self):
        lung_cancer_images = self.df[self.df[self.lung_cancer_label] ==
                self.true_label]['imageCount'].sum()
        not_lung_cancer_images = self.df[self.df[self.lung_cancer_label] ==
                self.false_label]['imageCount'].sum()
        total = self.df['imageCount'].sum()
        return (lung_cancer_images, not_lung_cancer_images, total)

    def _load_image(self, index, df, delete_file=False):
        from os.path import join
        l = df[(df['cumImageCount'] > index) & (df['imageCount'] != 0)]
        image_r = l.iloc[0]
        image_dir = join(self.data_dir, str(image_r[PID_LABEL]))
        img_ind = image_r['cumImageCount'] - index - 1
        assert(img_ind < image_r['imageCount'])
        files = os.listdir(image_dir)
        image_path = join(image_dir, files[int(img_ind) % len(files)])
        img = load_image(image_path, use_dicom=True)
        if delete_file:
            os.remove(image_path)
        return img, str(image_r[PID_LABEL])

    def train_data_generator(self):
        return self._retrieve_image_generator(self.df, self._t_tf, start_ind=0)

    def _retrieve_image_generator(self, df, tf_arr, start_ind=0):
        true_df = df[df[self.lung_cancer_label] == self.true_label].copy()
        true_df['cumImageCount'] = true_df['imageCount'].cumsum()
        false_df = df[df[self.lung_cancer_label] == self.false_label].copy()
        false_df['cumImageCount'] = false_df.loc[:, 'imageCount'].cumsum()
        true_ind = start_ind
        false_ind = start_ind
        for i in tf_arr:
            try:
                if i == 0:
                    img = self._load_image(
                            self._permuted_true_inds[true_ind], true_df, True)
                    true_ind += 1
                else:
                    img = self._load_image(
                            self._permuted_false_inds[false_ind], false_df, True)
                    false_ind += 1
                yield (img, i)
            except TypeError:
                continue
            except timeout_decorator.timeout_decorator.TimeoutError:
                continue

    def cv_data_generator(self):
        return self._retrieve_image_generator(
                self.df, self._cv_tf, start_ind=self._train_size)


def reduce_no_cancer_data(data_dir=DATA_DIR, csv_file=CSV_FILE, percent=0.50):
    '''
        Assumes directories are all already flattened!
        Warning: will delete a lot of your stuff
    '''
    df = pd.read_csv(csv_file)
    dirs = os.listdir(data_dir)
    for d in dirs:
        if df[df[PID_LABEL] == int(d)].iloc[0][LUNG_CANCER_LABEL] == FALSE_LABEL:
            print("Pruning %s" % d)
            # prune half of the files in the dir
            dirname = os.path.join(data_dir, d)
            files = listfiles(dirname)
            random.shuffle(files)
            delete_files = int(round(percent * len(files)))
            for f in files[:delete_files]:
                os.remove(os.path.join(dirname, f))


def prepare_data(cv_split=0.1, block_size=1024, img_size=(512,512),
        dest_dir=FAST_DATA_DIR):
    '''
        Resizes all images to 256x256 and moves them into a separate directory
    '''
    dl = DataLoader(refresh_counts=True)
    image_list = []
    label_list = []
    g = dl.train_data_generator()
    n = 497
    img_count = 0
    while True:
        assert(len(image_list) == len(label_list))
        if len(image_list) >= block_size:
            print("saving train (%d)" % n)
            img_arr = np.array(image_list)[:,:,:,None]
            label_arr = np.array(label_list)
            np.save(os.path.join(dest_dir, "train", "images", str(n)), img_arr)
            np.save(os.path.join(dest_dir, "train", "labels", str(n)), label_arr)
            n += 1
            image_list = []
            label_list = []
        try:
            print("loading image: %d" % img_count)
            (img, dirs), lbl = next(g)
            resized_img = scipy.misc.imresize(img, img_size)
            image_list.append(resized_img)
            label_list.append(int(dirs))
        except StopIteration:
            break
        except ValueError:
            continue

    n = 0
    g = dl.cv_data_generator()
    while True:
        assert(len(image_list) == len(label_list))
        if len(image_list) >= block_size:
            print("saving cv")
            img_arr = np.array(image_list)[:,:,:,None]
            label_arr = np.array(label_list)
            np.save(os.path.join(dest_dir, "cv", "images", str(n)), img_arr)
            np.save(os.path.join(dest_dir, "cv", "labels", str(n)), label_arr)
            n += 1
            image_list = []
            label_list = []
        try:
            (img, dirs), lbl = next(g)
            image_list.append(scipy.misc.imresize(img, img_size))
            label_list.append(int(dirs))
        except StopIteration:
            break


if __name__ == '__main__':
    prepare_data(block_size=16)
    # dl = DataLoader(refresh_counts=False)
    # g = dl.train_data_generator()
    # print(dl.get_statistics())
    # n = 0
    # while True:
        # img = next(g)
        # plt.subplot(121)
        # plt.imshow(img[0])
        # plt.subplot(122)
        # plt.imshow(scipy.misc.imresize(img[0], (256, 256)))
        # plt.show()
        # n += 1
        # print("\r%d" % n, end="")
        # print(img[0].shape)
