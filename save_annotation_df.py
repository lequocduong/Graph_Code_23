# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:39:34 2021

@author: zwg
"""
import rasterio
import pandas as pd
import os

fields_data_fpath = 'pinnote_anomaly_info/annotations_dates_params.csv'
fields_data = pd.read_csv(fields_data_fpath, sep=';')
sfd_ids = fields_data['sfd_id'].unique()
save_path_df = 'variables/annotation_df/'

for sfd_id_choice in sfd_ids:
    year_choice = fields_data[fields_data.sfd_id == sfd_id_choice].year.unique()[0]    
    image_names = os.listdir(f'pinnote_anomaly_info/annotations/{sfd_id_choice}/')
    image_names = [el for el in image_names if el.endswith('.tif')]
    annotation_df = []
    dates = []
    for el in image_names:
        dates.append(el.split('.')[0].split('_')[2])
        annotation_numpy = rasterio.open(f'pinnote_anomaly_info/annotations/{sfd_id_choice}/{el}').read(1, masked=True)
        annotation_df.append([annotation_numpy])
    annotation_df = pd.DataFrame(annotation_df, index=pd.to_datetime(dates, format='%Y-%m-%d'), columns=['annotes_numpy'])
    
    # # # save data for future analysis
    # if not os.path.exists(save_path_df):
    #     os.makedirs(save_path_df)
    # annotation_df.to_pickle(save_path_df + str(sfd_id_choice) + '_' + str(year_choice) + '_annotation_df.pkl')
        