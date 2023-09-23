# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:21:08 2021

@author: zwg
"""

import cv2
import scipy.ndimage as nd
import numpy as np

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)
    Output: 
        Return a filled array. 
    """

    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def resample(raster_in, raster_ref):
    raster_in_resampled = cv2.resize(fill(raster_in.filled(np.nan)), dsize=raster_ref.shape[::-1], interpolation=cv2.INTER_CUBIC)
    raster_in_resampled = np.ma.array(raster_in_resampled, mask=raster_ref.mask)
    return raster_in_resampled