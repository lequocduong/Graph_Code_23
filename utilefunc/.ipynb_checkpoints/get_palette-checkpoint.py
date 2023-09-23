# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:02:54 2021

@author: zwg
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# fonction definition
def get_palette(data):
    classes = []
    # get extreme classes
    m = np.nanmean(data)
    s = np.nanstd(data)
    mini = np.nanmin(data)
    maxi = np.nanmax(data)
    
    begin = []
    #### Compare the max-min of its distribution and set the boundary
    # 95 % of values within 2 Std
    if mini < (m-2*s):
        begin += [mini, (m-2*s)]
    else:
        begin += [mini]
    end = []
    if maxi > (m+2*s):
        end += [(m+2*s), maxi]
    else:
        end += [maxi]
    # print(begin, end)
    remain_len = 13 - len(begin) - len(end) + 2
    # linspace with the remaining
    remain = np.linspace(max(begin), min(end), remain_len)
    # print(remain)
    remain = remain[1:-1]
    
    bounds = begin + list(remain) + end
    cmap = mpl.colors.ListedColormap(np.array([
        [0,102,0,],
        [0,125,0,],
        [3,158,0,],
        [86,170,0,],
        [131,193,0,],
        [181,218,0,],
        [241,248,0,],
        [245,217,0,],
        [232,165,0,],
        [221,118,0,],
        [207,64,0,],
        [190,0,0,],
        [165,0,0,],
    ])[::-1,:]/255)
    #cmap.set_over('0.25')
    #cmap.set_under('0.75')
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

# display with palette:

# cmap, norm = get_palette(delivered_lai)
# plt.imshow((delivered_lai), cmap=cmap, norm=norm)