# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:00:10 2021

@author: zwg
"""

import os
import rasterio
import numpy as np

def data_extractor(data_path, sfd_id_choice, year_choice):
    """
    Extract and save file names for band image tif, band image gil and ndvi image tif 
    of a given field after filtering repeated images
    
    Parameters
    ----------
    data_path : string
        relative path for the images files
    sfd_id_choice : int or string
        season field id
    year_choice : int or string
        year. example : 2020
        
    Returns
    -------
    band_gil_file : string list
        list with all band image .gil file names
    band_tif_file : string list
        list with all band image .tif file names
    ndvi_tif_file : string list
        list with all ndvi image .tif file names
    """
    all_names = os.listdir(data_path + str(sfd_id_choice) + '/')
    image_names = [el for el in all_names if el.split('_')[2][:4] == str(year_choice)]
    image_names = sorted(image_names)
    
    band_tif_file= ['None']
    band_gil_file= ['None']
    ndvi_tif_file= ['None']
    # Problem 1 : There exist many images from the same satellite sensor and acquired for the same date, like : 
    # 103611228_Ndvi_20210526_Landsat8_Clear_13  103611228_Ndvi_20210526_Landsat8_Clear_14
    # we save the first one
    for img in image_names:
        if img.split('_')[1] == 'Ndvi' and img.split('_Clear_')[0] != ndvi_tif_file[-1].split('_Clear_')[0]:
            ndvi_tif_file.append(img)
        elif img.split('_')[1] == 'Bands' and img.endswith('.tif') and img.split('_Clear_')[0] != band_tif_file[-1].split('_Clear_')[0]:
            band_tif_file.append(img)
        elif img.split('_')[1] == 'Bands' and img.endswith('.gil') and img.split('_Clear_')[0] != band_gil_file[-1].split('_Clear_')[0]:
            band_gil_file.append(img)
    band_gil_file.pop(0)
    band_tif_file.pop(0)
    ndvi_tif_file.pop(0)
    
    # Problem 2 : many images of the same date but from different satellites :
    # '104196913_Bands_20210429_Landsat8_Clear_11.gil',  '104196913_Bands_20210429_Sentinel2_Clear_12.gil'
    # we choose Sentinel2 according to the priority
    
    todelete = []
    
    for i in range(1, len(band_gil_file)):
        if band_gil_file[i-1].split('_')[2] == band_gil_file[i].split('_')[2]:
            if band_gil_file[i-1].split('_')[3] != 'Sentinel2':
                todelete.append(i-1)
            elif band_gil_file[i].split('_')[3] != 'Sentinel2':
                todelete.append(i)                
    
    band_gil_file = np.delete(band_gil_file, todelete, 0)
    band_tif_file = np.delete(band_tif_file, todelete, 0)
    ndvi_tif_file = np.delete(ndvi_tif_file, todelete, 0)
    
    return band_gil_file, band_tif_file, ndvi_tif_file

def resize_ndvi(data_path, ndvi_tif_file, sfd_id_choice):
    """
    Preparation for ndvi image resizing
    
    Parameters
    ----------
    data_path : string
        relative path for the images files
    ndvi_tif_file : string list
        list with all ndvi image .tif file names
    sfd_id_choice : int or string
        season field id

    Returns
    -------
    mask_dict : dictionary
        dictionary with the key for season field id and the value for an numpy masked array
        this numpy masked array is of the standard size for next step, all unmasked values are 1
        the key is the season field id
    tif_info : dictionary
        this dictionary saves the EPSG and geotransform information of the .TIF file that corresponds
        to the standard image size
        the keys are 'EPSG' and 'transform'
    """
    # finding the most frequent size of NDVI images, consider it as the standard size
    h_w = [] # height-width
    for i in range(len(ndvi_tif_file)):
        img_ndvi_tif = data_path + str(sfd_id_choice) + '/' + ndvi_tif_file[i]  
        raster_ndvi = rasterio.open(img_ndvi_tif)
        h_w.append((raster_ndvi.height, raster_ndvi.width))
    h_w_norm = max(set(h_w), key=h_w.count)  # most frequent size

    # Create mask for the field : 
    mask_dict = dict()
    tif_info = dict()
    i=0
    while i < len(ndvi_tif_file) and i != -99:  # search for a standard size NDVI image
        img_ndvi_tif = data_path + str(sfd_id_choice) + '/' + ndvi_tif_file[i]  
        raster_ndvi = rasterio.open(img_ndvi_tif)
    
        if raster_ndvi.shape == h_w_norm:
            tif_info['EPSG'] = raster_ndvi.crs.to_epsg()
            tif_info['transform'] = raster_ndvi.transform
            mask = raster_ndvi.read(1, masked=True)
            mask.fill(1)
            mask_dict[sfd_id_choice] = mask  # values are replaced by 1, we need the standard image mask
            i = -99
        else :
            i=i+1
    
    return mask_dict, tif_info
