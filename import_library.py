import rasterio
import os, time, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from dtw import *
from skimage.segmentation import felzenszwalb
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
import cv2
import math
from scipy.ndimage import uniform_filter
from datetime import datetime
from pygeosys.timeserie.smoothers import  whitw
import joblib
import warnings

warnings.filterwarnings("ignore") #Hide messy numpy warnings
pd.options.display.float_format ='{:.2f}'.format