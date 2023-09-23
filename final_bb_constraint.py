# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:06:49 2021

@author: zwg
"""

#----------------------#
# cf. https://github.com/ekalinicheva/Unsupervised-CD-in-SITS-using-DL-and-Graphs
#     /Set_covering/final_bb_constraint.py
#----------------------#

import numpy as np

def final_bb_constraint(bb_final_list, segments_test, t1=0.4, t2=0):
    """
    Construct evolution graphs for each BB in the former selected set, it also eliminate BBs which are 
    less representatives
    
    Parameters
    ----------
    bb_final_list : a nx4 numpy array
        it stocks all necessary informations for the BBs
        n rows : n BBs
        4 columns : 
            1 - image position of a given BB:
            2 - corresponding segment ID in this image; 
            3 - number of pixels of a given BB;
            4 - novelty of a given BB
    segments_test : Pandas series
        time series of numpy array for the segmentation results; 
        index: date like '2021-04-02'
        value: numpy array of the same size as band image, number corresponds to segment id
    t1 : float, optional
        constraint the percentage of pixels of a segment inside the given BB 
        The default is 0.4.
    t2 : float, optional
        constraint the percentage of pixels of a BB inside the given segment
        The default is 0.

    Returns
    -------
    bb_final_list : a mx6 numpy array
        it stocks all necessary informations for the BBs as well as the attached segments (for evolution graph)
        m rows : m BBs finally choosen
        6 columns : 
            1 - image position of a given BB:
            2 - corresponding segment ID in this image; 
            3 - number of pixels of a given BB;
            4 - weight of a given BB;
            5 - N segments attached to the evolution graph of a given BB, 
                it is a Nx2 array where each column means (image_position, segment_ID) for each segment;
            6 - a Nx4 array where each column means, for each segment attached,  
                (image_position, segment_ID, % of pixels inside the given BB, the given BB's index in the (bb_final_list) 
    """
    # Constraint final BB
    t3 = 0#0.2    # tau3  # used in the paper, we don't use it 
    
    # try constructing graph by descending order of BB size
    bb_final_list = bb_final_list[np.flip(bb_final_list[:, 2].argsort(), axis=0)] 

    bb_final_list = np.c_[bb_final_list, np.full((len(bb_final_list), 2), None, dtype=object)]   # We add 2 empty columns
    
    # We open segmented images and construct candidates bb list.
    segm_array_list = []    # future stack of segmented images
    date_list = []  # image dates
    outliers_total_list = []
    outliers_total = None
    image_name_segm_list = segments_test.index.values
    nbr_images = len(image_name_segm_list)
    for i in range(nbr_images):
        date = image_name_segm_list[i]
        date_list.append(date)
        image_array_seg = segments_test.iloc[i]
        (H, W) = image_array_seg.shape
    
        segm_array_list.append(image_array_seg)
        outliers_array = np.zeros((H, W))
        
        if outliers_total is None:
            outliers_total = np.zeros((H, W))
        outliers_total += outliers_array
        
        outliers_total_list.append(outliers_array)
    
    # # #we create a grid
    covered_grids = np.copy(outliers_total_list) -1 # -1 not coverd grid, >=0 covered
    covered_grids_flatten = np.reshape(covered_grids, (nbr_images, H*W))  
    covered_grids_flatten_to_be_filled = np.copy(covered_grids_flatten)     # These grids represent the whole covered/incovered area of the dataset, -1 not covered, 1 covered
    covered_grids_flatten_to_be_filled_by_bb = np.zeros((nbr_images, 2, H*W))   #these grids will be filled with graphs, each graph is caracterized by corresponding BB id (img_id, seg_id)
    

    def create_graph_connection_weight(im_neigh, covered_ind_bb, size_bb, up_dir, connected_segments_ind_list, new_coverage_ind, connected_segments_ind_connection_weight_list, c, previously_constructed_graphs, previously_constructed_graphs_weights):
        '''
        This function attaches objects to a graph at each timestamp. It is directly in the code, so we can keep some parameters global.
        -   im_neigh is the timestamp that we check to add objects to the graph
        -   covered_ind_bb - footprint of the BB (pixels' ids)
        -   size_bb - size of bb
        -   up_dir - boolean, define whether the next timestamp to explore is t+1 or t-1 (we firstly explore all t+1 timestamps)
        -   connected_segments_ind_list - the list with the segments already attached to the bb
        -   new_coverage_id - footprint of the previously explored timestamp (pixels' ids)
        -   connected_segments_ind_connection_weight_list - the list with the segments already attached to the bb and their weights
        -   c - the BB's index in the list of BBs
        -   previously_constructed_graphs - list of segments that are already attached to other graphs
        -   previously_constructed_graphs_weights - list of segments that are already attached to other graphs and their weights,
            each element of the list contains [img_id, seg_id, weight, c]
        '''
        image_array_seg_flatten = segm_array_list[im_neigh].flatten()   #we open the image for the timestamp im_neigh (next or prior to bb)
        connected_segments = image_array_seg_flatten[new_coverage_ind]  # having the same pixel ids covered by bb, not necessarily all segment ids
        segments_to_check = np.unique(connected_segments)   # the potential segments to attach at this timestamp
        size_total_timestamp_intersection = connected_segments.size   #np.count_nonzero(connected_segments)    # we count pixels of change segments (0 - non-change area) 
        # We check if the intersection with previous timestamp (t3) is big enough
        # In other words, we check that the totel of segments to explore at this timestamp cover ar least t3 % of previous timestamp footprint
        if size_total_timestamp_intersection / len(new_coverage_ind) >= t3 and (len(segments_to_check) >= 1): #> 1 or (len(segments_to_check) == 1 and segments_to_check[0] != 0)):
            new_intersection_ind = None     # list with the pixels' ids of the segments that are attached at this timestamp
            size_total_timestamp = 0
            for s in segments_to_check:
                size_total_timestamp += len(np.where(image_array_seg_flatten == s)[0]) # the same pixel ids coverd by bb in im_neigh, the corresponding segment ids, their entire coverage scale in im_neigh
    
            # We iterate through the candidate segments
            for s in segments_to_check:

                if previously_constructed_graphs is not None:   # we check if it has already been attached to other graphs
                    first_where = np.where(np.asarray(previously_constructed_graphs)[:, 0] == im_neigh)[0] #normal numpy tricks did not work, so I wrote this ugly piece of code
                    second_where = np.where(np.asarray(previously_constructed_graphs)[:, 1] == s)[0]
                    if len(first_where)>0 and len(second_where)>0:
                        ind_in_list = np.intersect1d(first_where, second_where)  # the previous graphs id where the current seg of im_neigh already existed
                    else:
                        ind_in_list = []
                else:
                    ind_in_list = []
    
                coverage_ind_s = np.where(image_array_seg_flatten == s)[0]  # we get the pixels' ids of the segment that have intersection with bb
                size_s = len(coverage_ind_s)    # size of the segment
    
                # We check the intersection level with bb
                intersection_ind = np.intersect1d(covered_ind_bb, coverage_ind_s)
                size_s_inside_bb = len(intersection_ind)
                pourcentage_inside_bb = size_s_inside_bb/size_s  # percentage of the pixels of the segment s inside bb
                pourcentage_size_s_of_previous_step = size_s/len(new_coverage_ind)   
                
                pourcentage_cover_bb = size_s_inside_bb/size_bb  # limited by t2 
    
                # We check the weights of this segment in other graphs (if was attached)  # the weight is the degree of overlap of s with other bb
                if len(ind_in_list)>0:
                    previous_weight = np.asarray(previously_constructed_graphs_weights)[ind_in_list[0]][2]    # % of s_in_bb in the previous graph where the current seg of im_neigh already existed
                    previous_c = int(np.asarray(previously_constructed_graphs_weights)[ind_in_list[0]][3])   # the bb index in bb_list to which the weight is computed
                else:
                    is_it_already_a_bb = covered_grids_flatten[im_neigh][coverage_ind_s]
                    if np.sum(is_it_already_a_bb)>=0:  #>=0 covered
                        previous_weight=100  # bb intersecting with itself
                    else:
                        previous_weight = 0
                pourcentage_size_s_of_bb = size_s/size_bb   

                
                # We check if t1 condition is fulfilled and this segment do not have higher weight in another graph
                if pourcentage_inside_bb>=t1 and pourcentage_cover_bb>=t2 and pourcentage_inside_bb > previous_weight:
                    if len(ind_in_list) > 0: # we delete it from another graph if it is there
                        id_to_delete = (np.intersect1d(np.where(bb_final_list[previous_c][4][:,0]==im_neigh)[0], np.where(bb_final_list[previous_c][4][:,1]==s)[0]))
                        # print("INDEX TO DELETE", id_to_delete)
                        # print(previous_c)
                        bb_final_list[previous_c][4] = np.delete(bb_final_list[previous_c][4], id_to_delete, axis=0)
                        bb_final_list[previous_c][5] = np.delete(bb_final_list[previous_c][5], id_to_delete, axis=0)
                        previously_constructed_graphs = np.delete(previously_constructed_graphs, ind_in_list[0], axis=0)
                        previously_constructed_graphs_weights = np.delete(previously_constructed_graphs_weights, ind_in_list[0], axis=0)
    
                    # We add this segment to the list of previously attached segments
                    if previously_constructed_graphs is None:
                        previously_constructed_graphs = np.asarray([[im_neigh, s]])   
                        previously_constructed_graphs_weights = np.asarray([[im_neigh, s, pourcentage_inside_bb, c]])
                    else:
                        previously_constructed_graphs = np.concatenate((previously_constructed_graphs, np.asarray([[im_neigh, s]])), axis=0)   #im_neigh, seg_id_in_im_neigh
                        previously_constructed_graphs_weights = np.concatenate((previously_constructed_graphs_weights,
                                                                       np.asarray([[im_neigh, s, pourcentage_inside_bb, c]])), axis=0)   #im_neigh, seg_id_in_im_neigh, %_of_seg_inside_this_BB, BB's index in the list of BBs
                        _, first_ind = np.unique(previously_constructed_graphs, return_index=True, axis=0)
                        previously_constructed_graphs=previously_constructed_graphs[np.sort(first_ind)]
                        previously_constructed_graphs_weights = previously_constructed_graphs_weights[np.sort(first_ind)]
    
                    # we attach segment to graph's list
                    connected_segments_ind_list.append([im_neigh, s])
                    connected_segments_ind_connection_weight_list.append([im_neigh, s, pourcentage_inside_bb, c])
                    # we put the pixels' ids of this segment in the footprint list
                    if new_intersection_ind is None:
                        new_intersection_ind = intersection_ind
                    else:
                        new_intersection_ind = np.concatenate((new_intersection_ind, intersection_ind))

            if new_intersection_ind is not None:
                new_intersection_ind = new_intersection_ind.flatten()
            if up_dir is True:
                im_neigh += 1
            else:
                im_neigh -= 1
    
            if 0 <= im_neigh < nbr_images and new_intersection_ind is not None:
                # we pass to objects of another timestamp
                return create_graph_connection_weight(im_neigh, covered_ind_bb, size, up_dir, connected_segments_ind_list, new_intersection_ind, connected_segments_ind_connection_weight_list, c, previously_constructed_graphs, previously_constructed_graphs_weights)
            else:
                return connected_segments_ind_list, connected_segments_ind_connection_weight_list, previously_constructed_graphs, previously_constructed_graphs_weights
        else:
            return connected_segments_ind_list, connected_segments_ind_connection_weight_list, previously_constructed_graphs, previously_constructed_graphs_weights
    
    
    # We may have some objects that are attached to several graphs (explained in the article).
    # We have to deal with it by computing weight of each object (its intersection level with BB footprint).
    previously_constructed_graphs = None # we write here the objects that have already been attached to different graphs
    previously_constructed_graphs_weights = None # we write here the objects that have already been attached to different graphs and compute their weights
    
    to_delete = []
    for c in range(len(bb_final_list)): # we iterate through BBs list
        candidate = bb_final_list[c]
        # print(c)
        image, ind, size, weight, _, _= candidate
        coverage_ind = np.where(segm_array_list[image].flatten() == ind)[0] # We get indicies of every pixel of BB footprint
        # We check if this BB is not already attached to another graph
        if previously_constructed_graphs is not None and len(np.intersect1d(np.where(previously_constructed_graphs[:, 0] == image)[0],
                              np.where(previously_constructed_graphs[:, 1] == ind)[0])) > 0:
            index_to_explore = np.intersect1d(np.where(previously_constructed_graphs[:, 0] == image)[0],
                                              np.where(previously_constructed_graphs[:, 1] == ind)[0])[0]
            img_bb, ind_bb, weight_link, c_bb = previously_constructed_graphs_weights[index_to_explore]
            img_bb, ind_bb, c_bb = int(img_bb), int(ind_bb), int(c_bb)

        else:
            weight_link = 0

    
        # If the weight of this BB is higher than his intersection with another BB:
        if weight > weight_link:

            connected_segments_ind_list = []
            connection_weight_list = []
            # We attach the objects that are located in the next timestamps
            # Recursive function
            # print(image)
            # print(nbr_images - 1)
            if image < nbr_images - 1:
                connected_segments_ind_list, connected_segments_ind_connection_weight_list, previously_constructed_graphs, previously_constructed_graphs_weights = create_graph_connection_weight(image + 1, coverage_ind, size, True, connected_segments_ind_list, coverage_ind, connection_weight_list, c, previously_constructed_graphs, previously_constructed_graphs_weights)

            # We attach the objects from previous timestamps
            # Recursive function
            if image > 0:
                connected_segments_ind_list, connected_segments_ind_connection_weight_list, previously_constructed_graphs, previously_constructed_graphs_weights = create_graph_connection_weight(image - 1, coverage_ind, size, False, connected_segments_ind_list, coverage_ind, connection_weight_list, c, previously_constructed_graphs, previously_constructed_graphs_weights)

            if previously_constructed_graphs is None:
                to_delete.append(c)
            # If the constructed graph is not empty, we add the graph objects to the BB list
            elif len(connected_segments_ind_list) > 0:
                bb_final_list[c][4] = np.unique(connected_segments_ind_list, axis=0)  # im_neigh, seg_id_in_im_neigh
                bb_final_list[c][5] = np.unique(connection_weight_list, axis=0)  #im_neigh, seg_id_in_im_neigh, %_of_seg_inside_this_BB, BB's index in the list of BBs
                connected_segments_ind_list = bb_final_list[c][4]

                # We fill the grids with these objects
                for neigh in connected_segments_ind_list:
                    im_s, ind_s = neigh
                    coverage_s = np.where(segm_array_list[im_s].flatten() == ind_s)[0]
                    covered_grids_flatten_to_be_filled[im_s][coverage_s] = 1      # pixels of seg in other timestamp covered by the graphe of BB will be marked 1, as seg only belong to one graph, 
                    covered_grids_flatten_to_be_filled_by_bb[im_s, :, coverage_s] = [image, ind]
                covered_grids_flatten[image][coverage_ind] = ind
                covered_grids_flatten_to_be_filled[image][coverage_ind] = 1   # pixels covered by BB in this timestamp marked 1
                covered_grids_flatten_to_be_filled_by_bb[image, :, coverage_ind] = [image, ind]

                # We delete this BB from a previously constructed graph if it already was there and if it has smaller weight  (as bb may be covered by another bb) and (weight > weight_link:)
                if len(np.intersect1d(np.where(previously_constructed_graphs[:, 0] == image)[0], np.where(previously_constructed_graphs[:, 1] == ind)[0])) > 0:
                    id_to_delete = (np.intersect1d(np.where(bb_final_list[c_bb][4][:, 0] == image)[0],  np.where(bb_final_list[c_bb][4][:, 1] == ind)[0]))
                    bb_final_list[c_bb][4] = np.delete(bb_final_list[c_bb][4], id_to_delete, axis=0)
                    bb_final_list[c_bb][5] = np.delete(bb_final_list[c_bb][5], id_to_delete, axis=0)
        # else:
            # print("ALREADY THERE")
    
#     print(to_delete)

    # We delete empty graphs from the list with associated BBs and we attach their BBs to other graphs
    # It happens if we have deleted all the objects with weak connections from certain graphs.
    all_neighbours = bb_final_list[:,4]
    to_delete = []
    for n in range(len(all_neighbours)):
        if all_neighbours[n] is None or len(all_neighbours[n])==0:  # No object was ever attached to the BB
            if all_neighbours[n] is None:
                to_delete.append(n)
            elif len(all_neighbours[n])==0: # BB had attached objects that were later re-attached to other graphs
                to_delete.append(n)
                image, ind, _, _, _, _ = bb_final_list[n]
                coverage_ind = np.where(segm_array_list[image].flatten() == ind)[0]
                potential_bb_list = []
                # We look for all possible candidates-graphs to attach this BB as a normal object
                for i in range(nbr_images):
                    if i != image:
                        potential_bb = np.unique(covered_grids_flatten[i][coverage_ind])
                        for b in potential_bb:
                            if b >= 0: # meaning covered
                                potential_bb_list.append([i, b])  # image, seg_id
                # We choose the graph with higher weight for this BB and we attach it, sort and modify the grids
                potential_bb_list_link_weight = []
                if len(potential_bb_list) > 0:
                    for bb in range(len(potential_bb_list)):
                        i, b = np.asarray(potential_bb_list[bb], dtype=int)
                        coverage_bb = np.where(segm_array_list[i].flatten() == b)[0]
                        intersection_percent = len(np.intersect1d(coverage_ind, coverage_bb))/len(coverage_ind)
                        # print(intersection_percent)
                        potential_bb_list_link_weight.append(intersection_percent)
                    covered_grids_flatten[image][coverage_ind] = -1  
                    potential_bb_list_link_weight = np.asarray(potential_bb_list_link_weight)
                    done = False
                    for index in np.argsort(potential_bb_list_link_weight):
                        img_bb, ind_bb = potential_bb_list[index]
                        # print(img_bb, ind_bb)
    
                        id_bb_to_modify = np.intersect1d(np.where(bb_final_list[:][0] == img_bb)[0], np.where(bb_final_list[:][1] == ind_bb)[0])
                        if len(id_bb_to_modify)>0 and bb_final_list[id_bb_to_modify][4] is not None and len(bb_final_list[id_bb_to_modify][4])>0:
                            bb_final_list[id_bb_to_modify][4] = np.concatenate((bb_final_list[id_bb_to_modify][4], [[image, ind]]), axis=0)
                            bb_final_list[id_bb_to_modify][5] = np.concatenate((bb_final_list[id_bb_to_modify][5], [[image, ind, np.max(potential_bb_list_link_weight)]]),
                                                                               axis=0)
                            covered_grids_flatten_to_be_filled_by_bb[image, :, coverage_ind] = [img_bb, ind_bb]
                            done = True
                        if done:
                            break
                    if done==False: 
                        covered_grids_flatten_to_be_filled[image][coverage_ind] = -1 # 0 
                        covered_grids_flatten_to_be_filled_by_bb[image, :, coverage_ind] = [0, 0]
                else:
                    covered_grids_flatten_to_be_filled[image][coverage_ind] = -1 # 0
                    covered_grids_flatten_to_be_filled_by_bb[image, :, coverage_ind] = [0, 0]
    
    bb_final_list = np.delete(bb_final_list, to_delete, axis=0)
    
    # try visualizing each graph by descending order of BB's weight rather than by size
#     bb_final_list = bb_final_list[np.flip(bb_final_list[:, 3].argsort(), axis=0)] 

    return bb_final_list
