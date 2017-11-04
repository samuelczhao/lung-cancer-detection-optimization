#! /usr/bin/env python
"""
    General useful data loading functions
"""

import timeout_decorator
import dicom
import PIL
import os
import numpy as np

@timeout_decorator.timeout(5)
def load_image(loc, use_dicom=False):
    ''' Retrieves numpy array of image based off image dir '''
    from PIL import Image
    if not os.path.isfile(loc):
        raise ValueError("Invalid filepath passed")

    if use_dicom:
        ds = dicom.read_file(loc, force=True)
        return np.array(ds.pixel_array).copy()
    else:
        return np.load(loc)
