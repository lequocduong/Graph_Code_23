# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:00:00 2021

@author: zwg
"""

#----------------------#
# cf. https://github.com/ekalinicheva/Unsupervised-CD-in-SITS-using-DL-and-Graphs
#     /Set_covering/construct_set.py
#----------------------#

import numpy as np

def construct_bb_set(segments_test, alpha=0.2, direction=1, min_bb_size=40):
    """
    Construct a set of bounding boxs 
    
    Parameters
    ----------
    segments_test : Pandas series
        time series of numpy array for the segmentation results; 
        index: date like '2021-04-02'
        value: numpy array of the same size as band image, number corresponds to segment id
    alpha : float, optional
        constraint for the BB's novelty (weight). The default is 0.2.
    direction : bool, optional
        decide whether select bb from the biggest candidate to the smallest.
        The default is 1 (big -> small).
    min_bb_size : int, optional
        constraint for the BB's minimum number of pixels. The default is 40.

    Returns
    -------
    bb_final_list : a nx4 numpy array
        it stocks all necessary informations for the BBs
        n rows : n BBs
        4 columns : 
            1 - image position of a given BB:
            2 - corresponding segment ID in this image; 
            3 - number of pixels of a given BB;
            4 - weight of a given BB

    """
    ## Bounding Box (BB) selection
    # We create a list where we will write candidate BB
    bb_candidates_list = None
    
    # We open segmented images and construct candidates bb list. When all the segmented images are iterated we will start to calculate weights.
    # The list columns are Image_Id, Segm_Id, Segm_Size, Segm_Weight
    segm_array_list = []    # future stack of segmented images
    date_list = []  # image dates
    
    # We filter out only corresponding raster segmentation images. The list is sorted by date.
    image_name_segm_list = segments_test.index.values
    nbr_images = len(image_name_segm_list)
    
    # We iterate through segmentation images to make a stack of them
    for i in range(nbr_images):
        date = image_name_segm_list[i]
        # if i!= 0:
        #     image_name_segm_prev = image_name_segm_list[i - 1]
        #     date_prev = image_name_segm_list[i - 1] 
        # if i != len(image_name_segm_list)-1:
        #     image_name_segm2 = image_name_segm_list[i + 1]
        #     date2 = image_name_segm_list[i + 1] 
        date_list.append(date)
        
        image_array_seg = segments_test.iloc[i]
        (H, W) = image_array_seg.shape
        segm_array_list.append(image_array_seg)
    
        # We write all the segments to candidate BB list ([image_id, seg_id, seg_size, seg_weight=0])
    #     unique, count = np.unique(image_array_seg.flatten()[np.where(image_array_seg.flatten() != 0)[0]], return_counts=True)  # only change areas are segmented, no change areas have 0 id value.
        unique, count = np.unique(image_array_seg.flatten(), return_counts=True)
        candidates_bb_image = np.transpose([np.full(len(unique), i), unique, count, np.full(len(unique), None)])  # [image_id, seg_id, seg_size, seg_weight=0]
        if bb_candidates_list is None:
            bb_candidates_list = candidates_bb_image
        else:
            bb_candidates_list = np.concatenate((bb_candidates_list, candidates_bb_image), axis=0)
    bb_candidates_list = np.asarray(bb_candidates_list, dtype=object)
    
    # decide how to selection BBs from the candidate list
    if direction == 1:
        order = np.flip(bb_candidates_list[:, 2].argsort(), axis=0) # sort the candidate BB in descending order of size
    else:
        order = bb_candidates_list[:, 2].argsort()  # sort the candidate BB in ascending order of size
    bb_candidates_list = bb_candidates_list[order]

    # we create a grid that we will fill with bb. 0 - not covered grid, >=1 - covered grid (if >1, there is overlapping)
    covered_grid_flatten = np.zeros((H, W)).flatten()

    # We iterate through the list of candidate BB and we compute their novelty (weight)
    for c in range(len(bb_candidates_list)):
        candidate = bb_candidates_list[c]
        image, ind, size = candidate[:3]  # image_id, seg_id, seg_size
        if size > min_bb_size :#and size <= 0.5*H*W:   # we optionally set the minimum (and maximum) size for a BB  
            coverage_ind = np.where(segm_array_list[image].flatten()==ind)[0]   # we get pixel indicies of the segment
            novelty_pixels = covered_grid_flatten[coverage_ind] # we get values from covered grid to find out whether this footprint is already covered
            novelty_size = len(np.where(novelty_pixels == 0)[0])    # number of pixels that are not covered by any BB yet
            novelty = novelty_size/size     # novelty value
            # To better understand, read the article
            if novelty == 1:
                bb_candidates_list[c, 3] = size # segment weight
                covered_grid_flatten[coverage_ind] += 1  # we fill the grid with covered pixels
            elif novelty >= alpha and novelty < 1:
                bb_candidates_list[c, 3] = novelty
                # we recompute the weight of other candidates as there is an intersection
                not_novelty_coverage_ind = np.intersect1d(coverage_ind, np.where(covered_grid_flatten >= 1)[0], assume_unique=True)  # overlapping pixel index
                covered_grid_flatten[coverage_ind] += 1  # we fill the grid with covered pixels
                for im in range(nbr_images):
                    if im != image:
                        image_array_seg_flatten = segm_array_list[im].flatten()
                        segments_to_modify = np.unique(image_array_seg_flatten[not_novelty_coverage_ind])
    
                        for s in segments_to_modify:
                        #   we check only BBs that have been already visited and we modify their weight --- :c
                            index_bb_to_modify = np.intersect1d(np.where(bb_candidates_list[:c, 0]==im)[0], np.where(bb_candidates_list[:c, 1]==s)[0])
                            if len(index_bb_to_modify) > 0:
                                index_bb_to_modify = index_bb_to_modify[0]
                                index_of_to_be_modified, size_of_to_be_modified = bb_candidates_list[index_bb_to_modify, 1:3]
                                coverage_ind_of_to_be_modified = np.where(image_array_seg_flatten == index_of_to_be_modified)[0]
                                if bb_candidates_list[index_bb_to_modify, 3] != 0:
                                    novelty_to_be_modified = len(np.where(covered_grid_flatten[coverage_ind_of_to_be_modified]==1)[0])/size_of_to_be_modified
                                    if novelty_to_be_modified >= alpha:
                                        bb_candidates_list[index_bb_to_modify, 3] = novelty_to_be_modified
                                    else:
                                        bb_candidates_list[index_bb_to_modify, 3] = 0
                                        covered_grid_flatten[coverage_ind_of_to_be_modified] -= 1  # as bb is not novel anymore, we take it off from the coverage grid
            else:
                bb_candidates_list[c, 3] = 0
        else:
            bb_candidates_list[c, 3] = 0

    
    
    # We sort candidates BBs by descending order
    bb_candidates_list_by_weight = np.copy(bb_candidates_list[np.flip(bb_candidates_list[:, 3].argsort(), axis=0)])
    # We create the final list with final BBs that have weight greater than alpha (in our case all weights < aplha correspond to 0)
    bb_final_list = bb_candidates_list_by_weight[bb_candidates_list_by_weight[:, 3] > 0]
    
    return bb_final_list

