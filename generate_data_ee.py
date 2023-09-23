import matplotlib.pyplot as plt
import numpy as np
# import IPython.display as disp
import geopandas as gpd
from geemap import geopandas_to_ee
import pandas as pd
import requests
import logging
import ee
import xarray as xr
import numpy as np
import rasterio
from datetime import datetime
from pygeosys.util.dataframe import chunk_dataframe
import click

# Constants
# -----------------------------------------------------------------------------

# Parameters for cloud masking
#CLOUD_FILTER = 75
CLOUD_FILTER = 40
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

# Year of extraction
year=2022


# Landsat 8 NDVI functions

def clearNDVI_L57(image):
  ndvi = image.normalizedDifference(['B4', 'B3']).rename('NDVI');
  # CloudMask
  qa = image.select('pixel_qa');
  # If the cloud bit (5) is set and the cloud confidence (7) is high
  # or the cloud shadow bit is set (3), then it's a bad pixel.
  cloud = qa.bitwiseAnd(1 << 5).Or(qa.bitwiseAnd(1 << 3));
#    Remove edge pixels that don't occur in all bands
  mask = image.mask().reduce(ee.Reducer.min());

  #  Quality mask
  rad = image.select('radsat_qa');
  qual = rad.bitwiseAnd(1 << 1); # 0: valid data

  return ndvi.updateMask(cloud.Not()).updateMask(mask).updateMask(qual.Not())

def clearNDVI_L8(image):

  ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');

  #  CloudMask
  cloudShadowBitMask = (1 << 3);
  cloudsBitMask = (1 << 5);
  #  Get the pixel QA band.
  qa = image.select('pixel_qa');
  #  Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).bitwiseAnd(cloudsBitMask).eq(0);

  return ndvi.updateMask(mask);#.updateMask(qual);



# Functions s2cloudness for cloud masking
# -----------------------------------------------------------------------------

# S2 collection with cloud probability
def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))\
        .select('B3', 'B4','B8','QA60','SCL')

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

# Add clouds band
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

# Add clouds shadows band
def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


# Functions NDVI & synthesis
# -----------------------------------------------------------------------------
# Define the NDVI function

"""
def getRGBCollection(studyArea,startDate,endDate):

      '''
      Landsat 5 :  1984 – 2011
      Landsat 7 :  1999 - Present
      Landsat 8 :  mid 2013 - Present
      Sentinel 2 : 2016 - Present
      '''

      s2_sr_cld_col = get_s2_sr_cld_col(studyArea, startDate, endDate)
      s2_sr_cld_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)

      rgb = s2_sr_cld_col.select('B4','B3','B2')

      return rgb
"""



def NDVI(image):
    return image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

def L8_NDVI(image):
    ndvi = image.normalizedDifference(['B5','B4']).rename('NDVI')


def getNDVICollection(studyArea,startDate,endDate):

      '''
      Landsat 5 :  1984 – 2011
      Landsat 7 :  1999 - Present
      Landsat 8 :  mid 2013 - Present
      Sentinel 2 : 2016 - Present (TOC or surface reflectance from 2018)
      '''

      band_red = 'B3';
      band_nir = 'B4';
      col = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterDate(startDate,endDate).filterBounds(studyArea).select(band_red, band_nir, 'pixel_qa', 'radsat_qa');
      ndvi1 = col.map(clearNDVI_L57);

      band_red = 'B4';
      band_nir = 'B5';
      col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(startDate,endDate).filterBounds(studyArea).select(band_red, band_nir, 'pixel_qa', 'radsat_qa');
      ndvi2 = col.map(clearNDVI_L8);


      s2_sr_cld_col = get_s2_sr_cld_col(studyArea, startDate, endDate)
      s2_sr_cld_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).map(NDVI)

      ndvi3 = s2_sr_cld_col.select('NDVI')

      return ndvi1.merge(ndvi2).merge(ndvi3);

def get_NDVI_synthesis_S2(collection) :
    ndvi_synthesis = collection.reduce(**
                                       {'reducer' : ee.Reducer.percentile([95,50]).combine(
                                           **{'reducer2':ee.Reducer.stdDev(), "sharedInputs": True})
                                       }).rename(['maxNDVI', 'medNDVI', 'stdNDVI']).unmask(0).multiply(255).uint8()
    return ndvi_synthesis



@click.command()
# @click.argument('spatial_vector_file', type=click.Path(exists=True), )
@click.argument('input_filepath', type=click.Path(exists=True), )
@click.argument('year', type=int, )
# @click.argument('output_folder', type=click.Path(file_okay=False, exists=True,))
def main(input_filepath) :
    gee_project_name = 'datascience-field-border-dev'
    service_account = 'datascience-field-border-dev@datascience-field-border-dev.iam.gserviceaccount.com'
    private_key_path = '../../../datascience-field-border-dev-ab12b0aa44eb.json'
    credentials = ee.ServiceAccountCredentials(service_account, private_key_path)
    ee.Initialize(credentials, project=gee_project_name)
    # ee.Initialize()
    logger = logging.getLogger()
    # -----------------------------------------------------------------------------

    # Define input parameters
    # -----------------------------------------------------------------------------
    chunk_size = 1

    startJulian = 0
    endJulian = 364
    startDate = ee.Date.fromYMD(year, 1, 1).advance(startJulian, 'day');
    endDate = ee.Date.fromYMD(year, 1, 1).advance(endJulian, 'day');

    # load input vector dataset
    logger.info('Loading FeatureCollection')
    geometry_collection = gpd.read_file(input_filepath)

    if chunk_size is None:
        geometry_collections = [geometry_collection]
    else:
        geometry_collections = list(chunk_dataframe(geometry_collection, chunk_size))

    logger.info(f'Chunks: {len(geometry_collections)}')
    filename_prefixes = []
    # loop on chunks
    for chunk_id, gc in enumerate(geometry_collections):
        logger.info(
            f'Processing chunk {chunk_id} / {len(geometry_collections)}')
        logger.info(
            f'Uploading FeatureCollection ({len(gc)} Features) on server side')
        feature_collection_ee = geopandas_to_ee(gc)

        logger.info('Creating Task')

        feature = feature_collection_ee

        ndvi_col=getNDVICollection(feature.geometry(),startDate,endDate)

        result = get_NDVI_synthesis_S2(ndvi_col)

        output_filename = "Descartes/Annotations/rasters/Europe/Taiga"+str(chunk_id)+'_3857'
        fullName = 'ASE annotation NDVI synthesis SOuth America'

        task = ee.batch.Export.image.toCloudStorage(**{
            'image': result.clip(feature.geometry()),
            'crs': 'EPSG:3857',
            'description': fullName,
            'bucket': 'gri_geosys',
            'fileNamePrefix': output_filename,
            'scale': 10,
            'region': feature.geometry(),
            'fileDimensions': 3072,
            'maxPixels': 15000000000,
            'skipEmptyTiles': True
        })
        task.start()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
