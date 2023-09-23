# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:05:29 2021

@author: zwg
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
import geopandas as gpd
from skimage.segmentation import felzenszwalb

import matplotlib.pyplot as plt
from preprocess import data_extractor, resize_ndvi
from utilefunc.get_band_order_GS2 import get_band_order_GS2
from utilefunc.resample import resample
from construct_bb_set import construct_bb_set
from final_bb_constraint import final_bb_constraint

import os
import operator

## segmentation parameters 
scale=50        # controls the number of produced segments as well as their size. Higher scale means less and larger segments. 
sigma=0          # diameter of a Gaussian kernel, used for smoothing the image prior to segmentation.
min_size=5      # minimum size of the segment

## BB selection and graph construction parameters
alpha = 0.1  # for BB selection
t1 = 0.4     # for segment attachment to a BB to construct a graph
t2 = 0
direction=1  # direction of BB selection, 0: from small to big; 1: from big to small

## general data informations
fields_data_fpath = 'pinnote_anomaly_info/annotations_dates_params.csv'
fields_data = pd.read_csv(fields_data_fpath, sep=';')
sfd_ids = fields_data['sfd_id'].unique()  # all season field ids to process
data_path = 'data_images_2017_2020/'  # path for image time series, each stocked in a file folder named with the sfd_id


field_size = dict()
for sfd_id_choice in sfd_ids:
    year_choice = fields_data[fields_data.sfd_id == sfd_id_choice].year.unique()[0]
    _, _, ndvi_tif_file = data_extractor(data_path, sfd_id_choice, year_choice)
    mask_dict, _ = resize_ndvi(data_path, ndvi_tif_file, sfd_id_choice)
    num_valid_pixels = np.sum(~mask_dict[sfd_id_choice].mask)
#     print(mask_dict[sfd_id_choice].shape, num_valid_pixels, num_valid_pixels/(mask_dict[sfd_id_choice].shape[0]*mask_dict[sfd_id_choice].shape[1]))
    field_size[sfd_id_choice] = num_valid_pixels
    
for sfd_id_choice in sfd_ids:
    year_choice = fields_data[fields_data.sfd_id == sfd_id_choice].year.unique()[0]
    annotation_df = pd.read_pickle('variables/annotation_df/'+str(sfd_id_choice)+'_'+str(year_choice)+'_annotation_df.pkl')
    
    scale_precision = dict()
    scale_recall = dict()
    scale_F = dict()
    scale_bb_num = dict()
    for scale in range(0,105,5):
        band_gil_file, band_tif_file, ndvi_tif_file = data_extractor(data_path, sfd_id_choice, year_choice)
        
        if band_gil_file.size >= 2:    
            # ---------- Step 2 : Image data preprocessing
            # prepare for ndvi image resizing, the 'mask_dict' stocks an unified image size for the time series
            mask_dict, tif_info = resize_ndvi(data_path, ndvi_tif_file, sfd_id_choice)
            
            raster_df = []
            date = []
            # for each image of the satellite image time series for the sfd_id
            todelete = []
            for i in range(len(band_gil_file)):
                if band_gil_file[i].split('_')[3] in ['Sentinel2', 'Landsat8']:
                    img_band_gil = data_path + str(sfd_id_choice)+ '/' + band_gil_file[i]
                    img_band_tif = data_path + str(sfd_id_choice) + '/' + band_tif_file[i]
                    img_ndvi_tif = data_path + str(sfd_id_choice) + '/' + ndvi_tif_file[i]  # Bands and NDVI images are by the same date order
            
                    date.append(band_gil_file[i].split('_')[2]) 
                    
                    # raster band
                    bands_green_red_nir = get_band_order_GS2(img_band_gil, ['green', 'red', 'nir'])
                    raster_band = rasterio.open(img_band_tif)
                    raster_band_numpy = raster_band.read(bands_green_red_nir, masked=True) # attention masked # channel order : (3, height, width)
            
                    # raster ndvi
                    raster_ndvi = rasterio.open(img_ndvi_tif)
                    raster_ndvi_numpy = raster_ndvi.read(1, masked=True)
                    # resize the NDVI images
                    raster_ndvi_numpy = resample(raster_ndvi_numpy, mask_dict[sfd_id_choice])   
            
                    # resize bands images according to the unified NDVI images
                    raster_resampled_1 = resample(raster_band_numpy[0,:,:], raster_ndvi_numpy).filled(np.nan)
                    raster_resampled_2 = resample(raster_band_numpy[1,:,:], raster_ndvi_numpy).filled(np.nan)
                    raster_resampled_3 = resample(raster_band_numpy[2,:,:], raster_ndvi_numpy).filled(np.nan)
                    raster_band_numpy_resampled = np.stack((raster_resampled_1, raster_resampled_2, raster_resampled_3), axis=0)
                    raster_band_numpy_resampled = np.ma.masked_invalid(raster_band_numpy_resampled) # raster_band_numpy resampled with mask # not filled()
            
            # ---------- Step 3 : Segmentation 
                    # parameter 'scale' is proportional to the number of valide pixels for an image
                    # num_valid_pixels = np.sum(~raster_band_numpy_resampled[0,:,:].mask)
                    # scale = num_valid_pixels//100
            
                    # segment each band image 
                    raster_band_numpy_seg = np.transpose(raster_band_numpy_resampled, (1,2,0)) # channel : (width, height, 3) or (width, height) ndarray for segmentation
                    raster_band_numpy_seg = raster_band_numpy_seg.filled(-1)
                    segments_fz = felzenszwalb(raster_band_numpy_seg, scale=scale, sigma=sigma, min_size=min_size)
            
                    raster_df.append([raster_band_numpy_seg, segments_fz, raster_ndvi_numpy])
                else:
                    todelete.append(i)
            band_tif_file = np.delete(band_tif_file, todelete, 0)
            raster_df = pd.DataFrame(raster_df, index=pd.to_datetime(date, format='%Y-%m-%d'), columns=['raster_band_numpy_seg', 'segments_fz', 'raster_ndvi_numpy']).sort_index() # sort_index because the date labels may not be in order
                        
            # # save data for future analysis
            # if not os.path.exists(save_path_df):
            #     os.makedirs(save_path_df)
            # raster_df.to_pickle(save_path_df + str(sfd_id_choice) + '_' + str(year_choice) + '_raster_seg_df.pkl')
            
            # ---------- Step 4 : Bounding Box selection
            segments_test = raster_df['segments_fz']
            raster_ndvi_numpy_test = raster_df['raster_ndvi_numpy']
            bb_final_list_1 = construct_bb_set(segments_test, alpha, direction)
            
            # ---------- Step 5 : Evolution graph construction            
            ## Constraint final BB and construct graphs
            bb_final_list = final_bb_constraint(bb_final_list_1, segments_test, t1, t2)
        
            ## delete BB of invalid data (masked) zone
            todelete = []
            for i in range(bb_final_list.shape[0]):
                date_choice = segments_test.index[bb_final_list[i,0]].strftime('%Y-%m-%d')
                raster_ndvi_numpy = raster_df.loc[date_choice, 'raster_ndvi_numpy']
                segments_fz = raster_df.loc[date_choice, 'segments_fz']
            
            ###### delete graphs related to areas of masked invalid pixels 
                if raster_ndvi_numpy.mask[segments_fz == bb_final_list[i,1]].size !=0 and sum(raster_ndvi_numpy.mask[segments_fz == bb_final_list[i,1]])/raster_ndvi_numpy.mask[segments_fz == bb_final_list[i,1]].size >= 0.9:
                    todelete.append(i)
            bb_final_list = np.delete(bb_final_list, todelete, 0)
            
            # save data for future analysis
            # if not os.path.exists(save_path_bb):
            #     os.makedirs(save_path_bb)
            # np.save(save_path_bb+str(sfd_id_choice)+'_'+str(year_choice)+'_alpha_'+str(alpha)+'_t1_'+str(t1)+'_t2_'+str(t2)+'_final_bb.npy', bb_final_list)
    
    
            criterion_1 = []
            criterion_2 = []
            criterion_3 = []
            for i in range(bb_final_list.shape[0]):
                dico_bb = dict()
                for el in bb_final_list[i,4]:
                    dico_bb[el[0]] = dico_bb.get(el[0], []) + [el[1]]
                dico_bb[bb_final_list[i,0]] = [bb_final_list[i,1]]
                dico_bb = dict(sorted(dico_bb.items(), key=operator.itemgetter(0)))
                commun_dates = segments_test.index[list(dico_bb.keys())].intersection(annotation_df.index)
                commun_dates_idx = segments_test.index.get_indexer(commun_dates)
                dates_in_graph = len(commun_dates)/len(annotation_df.index)*100
                graph_not_anomaly = (1-len(commun_dates)/len(dico_bb))*100
    #             print(f'anomaly dates covered in evolution graph: {round(dates_in_graph,2)}%')
    #             print(f'evolution graph dates not abnormal : {round(graph_not_anomaly,2)}%')
    #             print(f'BB date : {segments_test.index[bb_final_list[i,0]]}')
    #             print(f'BB date in anomaly period : {segments_test.index[bb_final_list[i,0]] in annotation_df.index}')
                
                precision_avg = []
                recall_avg = []
                F_measure_avg = []
                for idx, j in enumerate(commun_dates_idx):
                    bb_cover = np.ma.zeros(segments_test.iloc[j].shape)
                    for el in dico_bb[j]:
                        bb_cover[segments_test.iloc[j] == el] = 1
                    bb_cover.mask = raster_ndvi_numpy_test.iloc[j].mask
                    annotation = annotation_df[annotation_df.index == commun_dates[idx]]['annotes_numpy'][0]
                    TP = (bb_cover*annotation).sum()
                    FP = (bb_cover*(annotation == 0)).sum()
                    FN = ((bb_cover == 0)*annotation).sum()
                    precision = TP/(TP+FP) 
                    recall = TP/(TP+FN)
                    F_measure = 2*precision*recall/(precision+recall)
                    precision_avg.append(precision)
                    recall_avg.append(recall)
                    F_measure_avg.append(F_measure)
            #         print(f'{commun_dates[idx]}  Precision : {round(precision,2)}  Recall : {round(recall,2)}  F-measure : {round(F_measure,2)}')
                precision_avg = np.nansum(precision_avg)/len(annotation_df)
                recall_avg = np.nansum(recall_avg)/len(annotation_df)
                F_measure_avg = np.nansum(F_measure_avg)/len(annotation_df)
    #             print(f'BB_{i}  Precision_avg : {round(precision_avg,2)}  Recall_avg : {round(recall_avg,2)}  F-measure_avg : {round(F_measure_avg,2)}')
    #             print()
                if precision_avg > 0 and recall_avg > 0:
                    criterion_1.append(precision_avg)
                    criterion_2.append(recall_avg)
                    criterion_3.append(F_measure_avg)
    
            scale_precision[scale] = criterion_1
            scale_recall[scale] = criterion_2
            scale_F[scale] = criterion_3
            scale_bb_num[scale] = bb_final_list.shape[0]
            
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.boxplot(scale_precision.values(), labels=scale_precision.keys())
    data_x = [i for i,k in enumerate(scale_precision.keys()) if len(scale_precision[k]) > 0 for el in scale_precision[k]]
    data_y = [el for v in list(scale_precision.values()) if len(v) > 0 for el in v]
    for x, y in zip(data_x, data_y):
        plt.scatter(x+1, y, alpha=0.4, c='r')
    plt.xlabel('scale value')
    plt.ylabel('average precision of all annotations per BB')
    
    plt.subplot(2,3,2)
    plt.boxplot(scale_recall.values(), labels=scale_recall.keys())
    data_x = [i for i,k in enumerate(scale_recall.keys()) if len(scale_recall[k]) > 0 for el in scale_recall[k]]
    data_y = [el for v in list(scale_recall.values()) if len(v) > 0 for el in v]
    for x, y in zip(data_x, data_y):
        plt.scatter(x+1, y, alpha=0.4, c='r')
    plt.xticks(range(1,len(scale_recall)+1), scale_recall.keys())
    plt.xlabel('scale value')
    plt.ylabel('average recall of all annotations per BB')
    
    plt.subplot(2,3,3)
    plt.boxplot(scale_F.values(), labels=scale_F.keys())
    data_x = [i for i,k in enumerate(scale_F.keys()) if len(scale_F[k]) > 0 for el in scale_F[k]]
    data_y = [el for v in list(scale_F.values()) if len(v) > 0 for el in v]
    for x, y in zip(data_x, data_y):
        plt.scatter(x+1, y, alpha=0.4, c='r')
    plt.xticks(range(1,len(scale_F)+1), scale_F.keys())
    plt.xlabel('scale value')
    plt.ylabel('average F-measure of all annotations per BB')
    
    plt.subplot(2,3,4)
    plt.grid(True)
    data = {key:sum(val) for key, val in scale_recall.items()}
    plt.plot(list(data.keys()), list(data.values()))
    plt.xticks(list(scale_recall.keys()), list(scale_recall.keys()))
    plt.xlabel('scale value')
    plt.ylabel('sum(average recall of all annotations per BB)')
    
    plt.subplot(2,3,5)
    plt.grid(True)
    plt.plot(list(scale_bb_num.keys()), list(scale_bb_num.values()))
    plt.xticks(list(scale_recall.keys()), list(scale_recall.keys()))
    plt.xlabel('scale value')
    plt.ylabel('number of BBs')
    
    plt.subplot(2,3,6)
    plt.grid(True)
    data_z = [len(v)/scale_bb_num[k]*100 for k, v in scale_recall.items()]
    plt.plot(list(scale_bb_num.keys()), data_z)
    plt.xticks(list(scale_recall.keys()), list(scale_recall.keys()))
    plt.xlabel('scale value')
    plt.ylabel('percentage of BBs covering annotated pixels (%)')
    
    plt.suptitle(f'{sfd_id_choice} {year_choice} valid_pixel//100 = {field_size[sfd_id_choice]//100}')
    plt.tight_layout()
    
    save_path = 'image_results/parameter_scale/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(save_path+str(sfd_id_choice)+'.png', format='png')
    plt.show()
    plt.close('all')
    print(sfd_id_choice)