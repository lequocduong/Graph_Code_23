# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:15:25 2021

@author: zwg
"""

#----------------------#
# cf. https://github.com/ekalinicheva/Unsupervised-CD-in-SITS-using-DL-and-Graphs
#     /Set_covering/plot_graph_optimized.py
#----------------------#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import text
from matplotlib.lines import Line2D
import warnings
import os


def evolution_graph_ndvi(fld_id_choice, year_choice, bb_final_list, segments_test, raster_ndvi_numpy_test, alpha=0.2, t1=0.4, t2=0):
    """
    Visualize all evolution graphs of a given field by curves of NDVI 
    
    Parameters
    ----------
    fld_id_choice : int or string
        choosen field id.
    year_choice : int
        choosen year.
    bb_final_list : a mx6 numpy array
        it stocks all necessary informations for the BBs as well as the attached segments (for evolution graph)
        m rows : m BBs finally choosen
        6 columns : 
            1 - image position of a given BB:
            2 - corresponding segment ID in this image; 
            3 - number of pixels of a given BB;
            4 - novelty of a given BB;
            5 - N segments attached to the evolution graph of a given BB, 
                it is a Nx2 array where each column means (image_position, segment_ID) for each segment;
            6 - a Nx4 array where each column means, for each segment attached,  
                (image_position, segment_ID, % of pixels inside the given BB, the given BB's index in the (bb_final_list) 
    segments_test : Pandas series
        time series of numpy array for the segmentation results; 
        index: date like '2021-04-02'
        value: numpy array of the same size as band image, number corresponds to segment id
    raster_ndvi_numpy_test : Pandas series
        time series of numpy array for the ndvi images; 
        index: date like '2021-04-02'
        value: numpy masked array of ndvi values,the same size as band image
    alpha : float, optional
        constraint for the BB's novelty (weight). The default is 0.2.
    t1 : float, optional
        constraint the percentage of pixels of a segment inside the given BB 
        The default is 0.4.
    t2 : float, optional
        constraint the percentage of pixels of a BB inside the given segment
        The default is 0.
    Returns
    -------
    figure
    """
    bb_ids = bb_final_list[:,:2]
    
    _, ax1 = plt.subplots(figsize=(10,8))
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.7, box.height])
    plt.grid(True)
    # markers = ['o', 'v', '*', '^', 'x', '<', '>', '8', 's', 'p',  'h', 'H', 'D', 'd', 'P', '.']
    # markersizes = bb_final_list[:,2]
    legend = []

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        for ind_m, bb_id in enumerate(bb_ids):
            image, ind = bb_id
    
            # We open segmented images
            segm_array_list = []
            date_list = []
            image_name_segm_list = segments_test.index.values
            nbr_images = len(image_name_segm_list)
    
            for i in range(nbr_images):
                date = image_name_segm_list[i]
                date_list.append(date)
    
                image_array_seg = segments_test.iloc[i]
                (H, W) = image_array_seg.shape
    
                segm_array_list.append(image_array_seg)
    
            # We find the graph by BB's id
            bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                              np.where(bb_final_list[:, 1] == ind)[0])[0]
            graph_info = bb_final_list[bb_index]
            # the infos of BB are not presented in the graph, we will insert it following the time order of the time series
            if graph_info[5][:,0].min() >= image:
                insert_bb_pos = 0
            elif graph_info[5][:,0].max() <= image:
                insert_bb_pos = graph_info[5][:,0].size - 1
            else:
                insert_bb_pos = np.where(graph_info[5][:,0] <= (image - 1))[0][-1] + 1
            sorted_graph_unorg = np.insert(graph_info[5], [insert_bb_pos], [image, ind, 1, 0], axis=0)
            timestamps = np.unique(sorted_graph_unorg[:,0]).astype('int')
            nb_timestamps = len(timestamps)
    
    
            sorted_graph = None
            for t in timestamps:
                segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
                image_array_seg = segm_array_list[t]
                ind_list=[]
                dist_list = []
                for seg in segments_at_timestamp:
                    ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
                    mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
                    dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
                    ind_list.append(mean_ind)
                    dist_list.append(dist)
                sort_ind = np.argsort(dist_list)
                new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
                # we append sorted objects to a sorted graph
                if sorted_graph is None:
                    sorted_graph = new_sorted
                else:
                    sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
            sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 4))    
    
    
            ndvi_mean_time = []
            ndvi_std_time = []
            time_seg = []
            for n in range(nb_timestamps):
                t = timestamps[n]
                segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
                nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
                
                ndvi_mean_time_seg = []
                for s in range(nb_seg):
                    seg = segments_at_timestamp[s][:2]
                    
                    # time_seg.append(segments_test.index[int(seg[0])])
                    ndvi_mean_time_seg.append(np.nanmean(raster_ndvi_numpy_test.iloc[int(seg[0])][segments_test.iloc[int(seg[0])] == int(seg[1])].filled(np.nan)))
                if not np.isnan(np.nanmean(ndvi_mean_time_seg)):
                    ndvi_mean_time.append(np.nanmean(ndvi_mean_time_seg))
                    ndvi_std_time.append(np.nanstd(ndvi_mean_time_seg))
                    time_seg.append(segments_test.index[t])
            
            if len(time_seg) != 0:
                plt.errorbar(pd.to_datetime(time_seg), ndvi_mean_time, yerr=ndvi_std_time, elinewidth=2, capsize=2)# marker=markers[ind_m],  
                legend.append('Evolution graph by BB : ' + str(bb_id[0]) + '-' + str(bb_id[1]))
                # ax = plt.scatter(pd.to_datetime(time_seg), ndvi_mean_time_seg, marker=markers[ind_m])
            
            y_pos = np.nanmean(raster_ndvi_numpy_test.iloc[int(bb_id[0])][segments_test.iloc[int(bb_id[0])] == int(bb_id[1])].filled(np.nan))
            if not np.isnan(y_pos):
                t = text(pd.to_datetime(segments_test.index[int(bb_id[0])]), y_pos,
                          'BB : '+ str(bb_id[0]) + '-' + str(bb_id[1]))
            
#         plt.legend(legend)
        ax1.legend(legend,loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        # ax1.legend(legend, bbox_to_anchor=(0.75, 0.7), bbox_transform=plt.gcf().transFigure)
        plt.ylim([0,1])
        plt.ylabel('Mean value of NDVI by segment in an evolution graph')
        plt.setp(ax1.get_xticklabels(), rotation=90)
        img_name = str(fld_id_choice) + '_' + str(year_choice) + '_evolution_graph_ndvi_alpha_'+str(alpha)+'_t1_'+str(t1)+'_t2_'+str(t2)
        
        
        plt.title(img_name)
        # savepath = 'image_results/evolution_graph/scale_5_alpha_'+str(alpha)+'_t1_'+str(t1)+'_t2_'+str(t2)+'/'
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)
        # plt.savefig(savepath +img_name+'.png', format='png')
        plt.show()

def evolution_graph_to_synopsis(fld_id_choice, year_choice, bb_final_list, segments_test, raster_ndvi_numpy_test, alpha=0.2, t1=0.4, t2=0):
    """
    Calculate the synopsis for each evolution graph of a given field, we can also visualize these synopsis by the fonction evolution_graph_synopsis

    Parameters are the same as the function 'evolution_graph_ndvi'
    Return a list of synopsis 
           each synopsis is a tuple of 3 elements:
               1- a list of timestamps Timestamp('2021-04-02 00:00:00') of the synopsis
               2- a list of weighted ndvi for each timestamp
               3- a 2-element array standing for the BB of this synopsis
                   image position of the BB:
                   corresponding segment ID in this image;
                   
    """
    bb_ids = bb_final_list[:,:2]

    synopsis = []
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        for ind_m, bb_id in enumerate(bb_ids):
            image, ind = bb_id
    
            # We open segmented images.
            segm_array_list = []
            date_list = []
            image_name_segm_list = segments_test.index.values
            nbr_images = len(image_name_segm_list)
    
            for i in range(nbr_images):
                date = image_name_segm_list[i]
                date_list.append(date)
    
                image_array_seg = segments_test.iloc[i]
                (H, W) = image_array_seg.shape
    
                segm_array_list.append(image_array_seg)
    
            # We find the graph by BB's id
            bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                              np.where(bb_final_list[:, 1] == ind)[0])[0]
            graph_info = bb_final_list[bb_index]
            if graph_info[5][:,0].min() >= bb_id[0]:
                insert_bb_pos = 0
            elif graph_info[5][:,0].max() <= bb_id[0]:
                insert_bb_pos = graph_info[5][:,0].size - 1
            else:
                insert_bb_pos = np.where(graph_info[5][:,0]<=(bb_id[0]-1))[0][-1] + 1
            sorted_graph_unorg = np.insert(graph_info[5], [insert_bb_pos], [bb_id[0], bb_id[1], 1, 0], axis=0)
            timestamps = np.unique(sorted_graph_unorg[:,0]).astype('int')
            nb_timestamps = len(timestamps)
    
    
            sorted_graph = None
            for t in timestamps:
                segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
                image_array_seg = segm_array_list[t]
                ind_list=[]
                dist_list = []
                for seg in segments_at_timestamp:
                    ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
                    mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
                    dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
                    ind_list.append(mean_ind)
                    dist_list.append(dist)
                sort_ind = np.argsort(dist_list)
                new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
                # we append sorted objects to a sorted graph
                if sorted_graph is None:
                    sorted_graph = new_sorted
                else:
                    sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
            sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 4))    # all good now
    
    
            ndvi_weighted_time = []
            time_seg = []
            for n in range(nb_timestamps):
                t = timestamps[n]
                segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
                nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
    
                cover_pixel_time = 0
                ndvi_mean_time = 0
                for s in range(nb_seg):
                    seg = segments_at_timestamp[s][:2]
                    cover_pixel = np.sum(segments_test.iloc[int(seg[0])] == int(seg[1]))
                    cover_pixel_time += cover_pixel
                    ndvi_mean = np.nanmean(raster_ndvi_numpy_test.iloc[int(seg[0])][segments_test.iloc[int(seg[0])] == int(seg[1])].filled(np.nan))
                    ndvi_mean_time += cover_pixel*ndvi_mean
                ndvi_mean_time = ndvi_mean_time/cover_pixel_time
                if not np.isnan(np.nanmean(ndvi_mean_time)):
                    ndvi_weighted_time.append(np.nanmean(ndvi_mean_time))
                    time_seg.append(segments_test.index[t])
    

            y_pos = np.nanmean(raster_ndvi_numpy_test.iloc[int(bb_id[0])][segments_test.iloc[int(bb_id[0])] == int(bb_id[1])].filled(np.nan))
            # if it is nan, the BB is the masked area without valid data
            if not np.isnan(y_pos):
                synopsis.append((time_seg, ndvi_weighted_time, bb_id))
    return synopsis   

def evolution_graph_synopsis(fld_id_choice, year_choice, bb_final_list, segments_test, raster_ndvi_numpy_test, alpha=0.2, t1=0.4, t2=0):
    """
    Calculate the synopsis for each evolution graph of a given field and visualize these synopsis

    Parameters are the same as the function 'evolution_graph_ndvi'
    Return is a figure 
           
    """
    bb_ids = bb_final_list[:,:2]

    _, ax1 = plt.subplots(figsize=(10,8))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.7, box.height])
    plt.grid(True)
    legend = []
#     markers = ['o', 'v', '*', '^', 'x', '<', '>', '8', 's', 'p',  'h', 'H', 'D', 'd', 'P']
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        for ind_m, bb_id in enumerate(bb_ids):
            image, ind = bb_id
    
            # We open segmented images.
            segm_array_list = []
            date_list = []
            image_name_segm_list = segments_test.index.values
            nbr_images = len(image_name_segm_list)
    
            for i in range(nbr_images):
                date = image_name_segm_list[i]
                date_list.append(date)
    
                image_array_seg = segments_test.iloc[i]
                (H, W) = image_array_seg.shape
    
                segm_array_list.append(image_array_seg)
    
            # We find the graph by BB's id
            bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                              np.where(bb_final_list[:, 1] == ind)[0])[0]
            graph_info = bb_final_list[bb_index]
            if graph_info[5][:,0].min() >= bb_id[0]:
                insert_bb_pos = 0
            elif graph_info[5][:,0].max() <= bb_id[0]:
                insert_bb_pos = graph_info[5][:,0].size - 1
            else:
                insert_bb_pos = np.where(graph_info[5][:,0]<=(bb_id[0]-1))[0][-1] + 1
            sorted_graph_unorg = np.insert(graph_info[5], [insert_bb_pos], [bb_id[0], bb_id[1], 1, 0], axis=0)
            timestamps = np.unique(sorted_graph_unorg[:,0]).astype('int')
            nb_timestamps = len(timestamps)
    
    
            sorted_graph = None
            for t in timestamps:
                segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
                image_array_seg = segm_array_list[t]
                ind_list=[]
                dist_list = []
                for seg in segments_at_timestamp:
                    ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
                    mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
                    dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
                    ind_list.append(mean_ind)
                    dist_list.append(dist)
                sort_ind = np.argsort(dist_list)
                new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
                # we append sorted objects to a sorted graph
                if sorted_graph is None:
                    sorted_graph = new_sorted
                else:
                    sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
            sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 4))    # all good now
    
    
            ndvi_weighted_time = []
            time_seg = []
            for n in range(nb_timestamps):
                t = timestamps[n]
                segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
                nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
    
                cover_pixel_time = 0
                ndvi_mean_time = 0
                for s in range(nb_seg):
                    seg = segments_at_timestamp[s][:2]
                    cover_pixel = np.sum(segments_test.iloc[int(seg[0])] == int(seg[1]))
                    cover_pixel_time += cover_pixel
                    ndvi_mean = np.nanmean(raster_ndvi_numpy_test.iloc[int(seg[0])][segments_test.iloc[int(seg[0])] == int(seg[1])].filled(np.nan))
                    ndvi_mean_time += cover_pixel*ndvi_mean
                ndvi_mean_time = ndvi_mean_time/cover_pixel_time
                if not np.isnan(np.nanmean(ndvi_mean_time)):
                    ndvi_weighted_time.append(np.nanmean(ndvi_mean_time))
                    time_seg.append(segments_test.index[t])
                
            if len(time_seg) != 0:
                plt.plot(pd.to_datetime(time_seg), ndvi_weighted_time)#, marker=markers[ind_m])
                legend.append('Evolution graph by BB : ' + str(bb_id[0]) + '-' + str(bb_id[1]))
            
            y_pos = np.nanmean(raster_ndvi_numpy_test.iloc[int(bb_id[0])][segments_test.iloc[int(bb_id[0])] == int(bb_id[1])].filled(np.nan))
            if not np.isnan(y_pos):
                t = text(pd.to_datetime(segments_test.index[int(bb_id[0])]), y_pos,
                          'BB : '+ str(bb_id[0]) + '-' + str(bb_id[1]))

        # plt.legend(legend)
        ax1.legend(legend,loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.ylim([0,1])
        plt.ylabel('NDVI synopsis')
        plt.setp(ax1.get_xticklabels(), rotation=90)

        img_name = str(fld_id_choice) + '-' + str(year_choice) + ' evolution graph synopsis'
        plt.title(img_name)
        # savepath = 'image_results/evolution_graph/synopsis/'
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)
        # plt.savefig(savepath +img_name+'.png', format='png')
        plt.show()

def evolution_graph_profil(fld_id_choice, year_choice, bb_final_list, segments_test):
    """
    Visualize relations between segments of every two consecutive dates of 
    the evolution graph. Segments will be linked if they have intersections.

    Parameters have the same meaning as the function 'evolution_graph_ndvi'
    Return all evolution graph relationship figures for a given field

    """
    bb_ids = bb_final_list[:,:2]
    for ind_m, bb_id in enumerate(bb_ids):
    #     bb_id = [2, 3]     # BB's id [img_id, seg_id]
        image, ind = bb_id

        # We open segmented images.
        segm_array_list = []
        date_list = []
        image_name_segm_list = segments_test.index.values
        nbr_images = len(image_name_segm_list)
        
        for i in range(nbr_images):
            date = image_name_segm_list[i]
            date_list.append(date)
            
            image_array_seg = segments_test.iloc[i]
            (H, W) = image_array_seg.shape
        
            segm_array_list.append(image_array_seg)
        
        # We find the graph by BB's id
        bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                          np.where(bb_final_list[:, 1] == ind)[0])[0]
        graph_info = bb_final_list[bb_index]
        if graph_info[5][:,0].min() >= bb_id[0]:
            insert_bb_pos = 0
        elif graph_info[5][:,0].max() <= bb_id[0]:
            insert_bb_pos = graph_info[5][:,0].size - 1
        else:
            insert_bb_pos = np.where(graph_info[5][:,0]<=(bb_id[0]-1))[0][-1] + 1
        sorted_graph_unorg = np.insert(graph_info[5], [insert_bb_pos], [bb_id[0], bb_id[1], 1, 0], axis=0)
        timestamps = np.unique(sorted_graph_unorg[:,0]).astype('int')
        nb_timestamps = len(timestamps)
        
        
        sorted_graph = None
        for t in timestamps:
            segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
            image_array_seg = segm_array_list[t]
            ind_list=[]
            dist_list = []
            for seg in segments_at_timestamp:
                ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
                mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
                dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
                ind_list.append(mean_ind)
                dist_list.append(dist)
            sort_ind = np.argsort(dist_list)
            new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
            # we append sorted objects to a sorted graph
            if sorted_graph is None:
                sorted_graph = new_sorted
            else:
                sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
        sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 4))    # all good now
        
        
        
        
        graph_sceleton = [] # list with nbr of segments at each timestamp
        connections = []
        for t in timestamps:
            segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
            nb_seg = segments_at_timestamp.shape[0]   # nb of segments at this timestamp
            graph_sceleton.append(nb_seg)
            image_array_seg = segm_array_list[t].flatten()
            if t < max(timestamps):
                image_array_seg_next = segm_array_list[t+1].flatten()
                # We look to the segments at next timestamp that are connected to this timestamp
                for seg in segments_at_timestamp:
                    ind = np.where(image_array_seg == seg[1])[0]
                    # intersection between seg pixel footprint of time t in time t+1 and seg id in time t+1, so get the seg ids at t+1 connected to each seg at t
                    connected = np.intersect1d(np.unique(image_array_seg_next[ind]), sorted_graph[:, 1][sorted_graph[:, 0] == t+1])  
        #             connected = np.intersect1d(np.setdiff1d(np.unique(image_array_seg_next[ind]), [0]), sorted_graph[:, 1][sorted_graph[:, 0] == t+1])
                    connections.append([seg[:2], connected])    # edge
        max_graphs = np.max(graph_sceleton) #the widest part of the graph
        connections = np.asarray(connections, dtype=object)
        # print(connections)
        
        
        # Parameters to draw the ellipses (nodes) that correspond to segments
        # Better not to touch
        ell_width = 0.45
        ell_height = 0.35
        space_width = 0.01
        space_height = 0.75
        fig_width = max_graphs * ell_width + space_width * (max_graphs - 1)
        fig_height = nb_timestamps * ell_height + space_height * (nb_timestamps - 1)
        
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        ell_width = ell_width/fig_width
        ell_height = ell_height/fig_height
        space_width = space_width/fig_width
        space_height = space_height/fig_height
        
        # We draw ellipces and we put segments numbers inside
        y_start = 1 - ell_height/2
        coordinates = []
        for n in range(nb_timestamps):
            t = timestamps[n]
            segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
            nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
            if nb_seg == max_graphs:
                x_start = ell_width/2
            else:
                diff_w_max = max_graphs - nb_seg
                x_start = (diff_w_max / 2 * (ell_width + space_width)) + (ell_width/2)
            for s in range(nb_seg):
                seg = segments_at_timestamp[s][:2]
                x = x_start + s * (ell_width + space_width)
                xy = [x, y_start]
                e = Ellipse(xy, ell_width, ell_height, angle=0)
                e.set_facecolor("white")
                e.set_edgecolor("black")
                ax.add_artist(e)
                coordinates.append([seg, xy])
                t = text(x, y_start, str(seg[0])+"-"+str(seg[1]), horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes,
                         fontname = 'Times New Roman')
                t.set_fontsize(7)
            if n == image:
                e.set_facecolor("white")
                e.set_edgecolor("red")
                t.set_weight("bold")
                t.set_color("red")
                t = text(x+ell_width, y_start, 'BB', horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes, 
                         fontname = 'Times New Roman', color='red', weight='bold')
            y_start -= (ell_height + space_height)
        coordinates = np.asarray(coordinates, dtype=object)
        
        
        
        # We draw the edges
        for n in range(len(connections)):
            c = connections[n]
            coord1 = coordinates[n][1]
        #     print(coord1)
            coord1 = coord1[0], coord1[1] - ell_height/2
            seg = c[0]
            connected_to = c[1]
            for seg_con in connected_to:
                coord2 = coordinates[np.intersect1d(np.where(coordinates[:, 0, 0] == seg[0]+1)[0], np.where(coordinates[:, 0, 1] == seg_con)[0])][0][1]
                coord2 = coord2[0], coord2[1] + ell_height / 2
        #         print([coord1[0], coord2[0]], [coord1[1], coord2[1]])
                l = Line2D([coord1[0], coord2[0]], [coord1[1], coord2[1]], lw=0.5, color="black")
                ax.add_line(l)
            coord1 = coord1[0], coord1[1] - ell_height/2
        
        plt.xticks([])
        plt.yticks([])
        
        img_name = str(fld_id_choice) + '_' + str(year_choice) + '_evolution_graph_bb_' + str(bb_id[0]) + '_' + str(bb_id[1])
        # savepath = 'image_results/evolution_graph/'+str(fld_id_choice)+'/'
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)
        # plt.savefig(savepath + img_name +'.png', format='png')
        plt.show()



def evolution_graph_profil_ndvi(fld_id_choice, year_choice, bb_final_list, segments_test, raster_ndvi_numpy_test):
    """
    Visualize relations between segments of every two consecutive dates of 
    the evolution graph. They will be linked if they have intersections.
    The position and the color of each segement depend on its NDVI value.

    Parameters are the same as the function 'evolution_graph_ndvi'
    Return all evolution graph relationship figures for a given field


    """
    bb_ids = bb_final_list[bb_final_list[:,3]<1][:,:2] #bb_final_list[:,:2]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        for ind_m, bb_id in enumerate(bb_ids):
        #     bb_id = [2, 3]     # BB's id [img_id, seg_id]
            image, ind = bb_id
    
            # We open segmented images.
            segm_array_list = []
            date_list = []
            image_name_segm_list = segments_test.index.values
            nbr_images = len(image_name_segm_list)
            
            for i in range(nbr_images):
                date = image_name_segm_list[i]
                date_list.append(date)
                
                image_array_seg = segments_test.iloc[i]
                (H, W) = image_array_seg.shape
            
                segm_array_list.append(image_array_seg)
            
            # We find the graph by BB's id
            bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                              np.where(bb_final_list[:, 1] == ind)[0])[0]
            graph_info = bb_final_list[bb_index]
            # the graph, the segments at the same timestamp are not in the good order "geographically".
            # it means that the nodes are not organized by geographical proximity to each other, and when we connect the nodes, the edges will look disorganised.
            # sorted_graph_unorg = graph_info[5]   # im_neigh, seg_id_in_im_neigh, %_of_seg_inside_this_BB, BB's index in the list of BBs
            if graph_info[5][:,0].min() >= bb_id[0]:
                insert_bb_pos = 0
            elif graph_info[5][:,0].max() <= bb_id[0]:
                insert_bb_pos = graph_info[5][:,0].size - 1
            else:
                insert_bb_pos = np.where(graph_info[5][:,0]<=(bb_id[0]-1))[0][-1] + 1
            
            sorted_graph_unorg = np.insert(graph_info[5], [insert_bb_pos], [bb_id[0], bb_id[1], 1, 0], axis=0)
            timestamps = np.unique(sorted_graph_unorg[:,0]).astype('int')
            nb_timestamps = len(timestamps)
            
            
            sorted_graph = None
            for t in timestamps:
                segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
                image_array_seg = segm_array_list[t]
                ind_list=[]
                dist_list = []
                for seg in segments_at_timestamp:
                    ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
                    mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
                    dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
                    ind_list.append(mean_ind)
                    dist_list.append(dist)
                sort_ind = np.argsort(dist_list)
                new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
                # we append sorted objects to a sorted graph
                if sorted_graph is None:
                    sorted_graph = new_sorted
                else:
                    sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
            # sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 2))    # all good now
            sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 4))    # all good now
            
            
            
            
            graph_sceleton = [] # list with nbr of segments at each timestamp
            connections = []
            for t in timestamps:
                segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
                nb_seg = segments_at_timestamp.shape[0]   # nb of segments at this timestamp
                graph_sceleton.append(nb_seg)
                image_array_seg = segm_array_list[t].flatten()
                if t < max(timestamps):
                    image_array_seg_next = segm_array_list[t+1].flatten()
                    # We look to the segments at next timestamp that are connected to this timestamp
                    for seg in segments_at_timestamp:
                        ind = np.where(image_array_seg == seg[1])[0]
                        # intersection between seg pixel footprint of time t in time t+1 and seg id in time t+1, so get the seg ids at t+1 connected to each seg at t
                        connected = np.intersect1d(np.unique(image_array_seg_next[ind]), sorted_graph[:, 1][sorted_graph[:, 0] == t+1])  
            #             connected = np.intersect1d(np.setdiff1d(np.unique(image_array_seg_next[ind]), [0]), sorted_graph[:, 1][sorted_graph[:, 0] == t+1])
                        connections.append([seg[:2], connected])    # edge
            max_graphs = np.max(graph_sceleton) #the widest part of the graph
            connections = np.asarray(connections, dtype=object)
            # print(connections)
            
            # Parameters to draw the ellipses (nodes) that correspond to segments
            # Better not to touch
            ell_width = 0.45
            ell_height = 0.35
            space_width = 0.01
            space_height = 0.75
            fig_width = max_graphs * ell_width + space_width * (max_graphs - 1)
            fig_height = nb_timestamps * ell_height + space_height * (nb_timestamps - 1)
            
            
            fig, ax = plt.subplots(figsize=(fig_width+3, fig_height))
            plt.grid(True)
            
            ell_width = ell_width/fig_width
            ell_height = ell_height/fig_height
            space_width = space_width/fig_width
            space_height = space_height/fig_height
            
            # We draw ellipces and we put segments numbers inside
            y_start = 1 - ell_height/2
            ytick_mark = []
            coordinates = []
            for n in range(nb_timestamps):
                t = timestamps[n]
                segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
                nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
                if nb_seg == max_graphs:
                    x_start = ell_width/2
                else:
                    diff_w_max = max_graphs - nb_seg
                    x_start = (diff_w_max / 2 * (ell_width + space_width)) + (ell_width/2)
                for s in range(nb_seg):
                    seg = segments_at_timestamp[s][:2]
                    ndvi_mean = np.nanmean(raster_ndvi_numpy_test.iloc[int(seg[0])][segments_test.iloc[int(seg[0])] == int(seg[1])].filled(np.nan))
                    # x = x_start + s * (ell_width + space_width)
                    x = ndvi_mean
                    xy = [x, y_start]
                    e = Ellipse(xy, ell_width, ell_height, angle=0)
                    e.set_facecolor(color=(255/255*ndvi_mean,0/255,0/255))
                    e.set_edgecolor("black")
                    ax.add_artist(e)
                    coordinates.append([seg, xy])
                    # t = text(x, y_start, str(seg[0])+"-"+str(seg[1]), horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes,
                    #          fontname = 'Times New Roman')
                    # t.set_fontsize(7)
                if n == image:
                    e.set_facecolor("white")
                    e.set_edgecolor("red")
                    # t.set_weight("bold")
                    # t.set_color("red")
                    # t = text(x+0.2, y_start, 'BB', horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes, 
                    #          fontname = 'Times New Roman', color='red', weight='bold')
                ytick_mark.append(y_start)
                y_start -= (ell_height + space_height)
            coordinates = np.asarray(coordinates, dtype=object)
            
            
            
            # We draw the edges
            for n in range(len(connections)):
                c = connections[n]
                coord1 = coordinates[n][1]
            #     print(coord1)
                coord1 = coord1[0], coord1[1] - ell_height/2
                seg = c[0]
                connected_to = c[1]
                for seg_con in connected_to:
                    coord2 = coordinates[np.intersect1d(np.where(coordinates[:, 0, 0] == seg[0]+1)[0], np.where(coordinates[:, 0, 1] == seg_con)[0])][0][1]
                    coord2 = coord2[0], coord2[1] + ell_height / 2
            #         print([coord1[0], coord2[0]], [coord1[1], coord2[1]])
                    l = Line2D([coord1[0], coord2[0]], [coord1[1], coord2[1]], lw=0.5, color="black")
                    ax.add_line(l)
                coord1 = coord1[0], coord1[1] - ell_height/2
            
            plt.xlabel('Mean value of NDVI in the segment')
            plt.yticks(ytick_mark, labels=segments_test.index[timestamps], fontsize=7)
            # plt.savefig("/home/user/Dropbox/article_Montpellier/figures/graph_plot.svg", format="svg")
            plt.show()
