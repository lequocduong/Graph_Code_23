import os
import subprocess
import pandas as pd
import glob
from pathlib import Path

# import gdal
from datetime import datetime
import random
# from osgeo import gdal
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from scipy import interpolate
import sklearn.metrics as metrics

#from shapely.geometry import Polygon


class crosscalibration:
    def __init__(
        self,
        path_tmp,
        outpath_plt,
        path_ref,
        sensor_to_calibrate,
        path_sensor,
        band_relat,
        item_date_sensor,
        path_coeff,
        band_ranking,
        gain_sensor,
        bias_sensor,
    ):
        self.path_tmp = Path(path_tmp)
        self.outpath_plt = Path(outpath_plt)
        self.ref_name = "Sentinel-2"
        self.gain_ref = 1.2 / 10000 #65535
        self.bias_ref = 0
        self.item_date_ref = -4
        self.path_ref = Path(path_ref)
        self.sensor_name = sensor_to_calibrate
        self.path_sensor = Path(path_sensor)
        self.band_relat = band_relat
        self.gain_sensor = gain_sensor
        self.bias_sensor = bias_sensor
        self.item_date_sensor = item_date_sensor
        self.path_coeff = Path(path_coeff)
        self.band_ranking = band_ranking

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1.left < bb1.right
        assert bb1.bottom < bb1.top
        assert bb2.left < bb2.right
        assert bb2.bottom < bb2.top

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1.left, bb2.left)
        y_top = min(bb1.top, bb2.top)
        x_right = min(bb1.right, bb2.right)
        y_bottom = max(bb1.bottom, bb2.bottom)

        if x_right < x_left or y_top < y_bottom:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        inter_area = (x_right - x_left) * (y_top - y_bottom)

        # compute the area of both AABBs
        bb1_area = (bb1.right - bb1.left) * (bb1.top - bb1.bottom)
        bb2_area = (bb2.right - bb2.left) * (bb2.top - bb2.bottom) 

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(bb1_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou * 100

    def scatter_cal(self, x, y, str_band, sensor, reference, outfileName, df, im):
        COLUMN_NAMES = {
            '%s'  %str_band + '_rmse',
            '%s'  %str_band + '_r2',
            '%s'  %str_band + '_diffm',
            '%s'  %str_band + '_diffmed',
            '%s'  %str_band + '_pourcoutrange',
            '%s'  %str_band + '_std',
            '%s'  %str_band + '_gain',
            '%s'  %str_band + '_offset',
            }      
        # to plot and save the scatter plot
        fig = plt.figure(figsize=(8, 8))

        idx_nonan = ~np.isnan(x) & ~np.isnan(y)
        m, b = np.polyfit(x[idx_nonan], y[idx_nonan], 1)
        std = np.nanstd(abs((m * x[idx_nonan] + b) - y[idx_nonan]))
        sigma2 = std*2
        idx_model = abs((m * x[idx_nonan] + b) - y[idx_nonan])<=sigma2
        idx_model_out = abs((m * x[idx_nonan] + b) - y[idx_nonan])>sigma2
        pourcoutrange = np.count_nonzero(abs((m * x[idx_nonan] + b)-y[idx_nonan])>sigma2)/len(abs((m * x[idx_nonan] + b)-y[idx_nonan])>sigma2)*100

        if np.count_nonzero(idx_model) > 10:
            plt.plot(x[idx_nonan][idx_model], y[idx_nonan][idx_model], ".",label='data analyzed')
            plt.plot(x[idx_nonan][idx_model_out], y[idx_nonan][idx_model_out], ".",label='out of range 2sigma')

            m, b = np.polyfit(x[idx_nonan][idx_model], y[idx_nonan][idx_model], 1)
            mse = metrics.mean_squared_error(x[idx_nonan][idx_model], m * x[idx_nonan][idx_model] + b)
            rmse = np.sqrt(mse) # or mse**(0.5)  
            r2 = metrics.r2_score(x[idx_nonan][idx_model],m * x[idx_nonan][idx_model] + b)

            diffm = np.nanmean(x[idx_nonan][idx_model]-y[idx_nonan][idx_model])
            diffmed = np.nanmedian(x[idx_nonan][idx_model]-y[idx_nonan][idx_model])
            plt.xlabel(sensor, fontsize=12)
            plt.ylabel(reference, fontsize=12)
            plt.title(f'{str_band} band\n gain={np.round(m,3)} offset={np.round(b,3)} \n std={np.round(std,3)}\
                rmse={np.round(rmse,3)} r2={np.round(r2,3)} \n diff mean={np.round(diffm,3)} diff med={np.round(diffmed,3)}\
                    \n %pixel out of range={np.round(pourcoutrange,3)}', fontsize=14)
            plt.plot(x[idx_nonan], x[idx_nonan], label="x=y")
            plt.plot(x[idx_nonan], m * x[idx_nonan] + b, label="model")
            plt.legend()
            plt.savefig(outfileName)
        else:
            rmse = np.nan
            r2 = np.nan
            diffm = np.nan
            diffmed = np.nan
            pourcoutrange = np.nan
            std = np.nan
            m = np.nan
            b = np.nan
        if not COLUMN_NAMES.issubset(df.columns):
            df = pd.concat([df, pd.DataFrame(columns=list(COLUMN_NAMES))])

        df['%s'  %str_band + '_rmse'][im] = np.round(rmse,3)
        df['%s'  %str_band + '_r2'][im] = np.round(r2,3)
        df['%s'  %str_band + '_diffm'][im] = np.round(diffm,3)
        df['%s'  %str_band + '_diffmed'][im] = np.round(diffmed,3)
        df['%s'  %str_band + '_pourcoutrange'][im] = np.round(pourcoutrange,3)
        df['%s'  %str_band + '_std'][im] = np.round(std,3)
        df['%s'  %str_band + '_gain'][im] = np.round(m,3)
        df['%s'  %str_band + '_offset'][im] = np.round(b,3)
        return df

    def resample_mask_by_average(self, x, b_size):
        x_reduced = block_reduce(
            x, block_size=(b_size, b_size), func=np.mean, cval=np.mean(x)
        )
        return x_reduced

    def resample_band_by_average(self, x, b_size, mask_sensor, mask_ref):
        x_reduced = self.resample_mask_by_average(x, b_size)
        x_reduced[mask_sensor != 1] = np.nan
        x_reduced[mask_ref != 1] = np.nan
        x_reduced[mask_sensor == 0] = np.nan
        x_reduced[mask_ref == 0] = np.nan
        return x_reduced
    
    def resample_band_by_average_WithoutMask(self, x, b_size):
        x_reduced = self.resample_mask_by_average(x, b_size)
        return x_reduced

    def get_list_dataset_sensor(self, struct_name):
        lst_images = glob.glob(os.path.join(self.path_sensor, struct_name))
        return lst_images

    def get_list_dataset_reference(self, struct_name):
        lst_images = glob.glob(os.path.join(self.path_ref, struct_name))
        return lst_images

    def test_read_date_sensor(self, lst_images):
        all_dates = []
        for im_name in lst_images:
            dateim = im_name.split("_")
            dateim = datetime.strptime(dateim[self.item_date_sensor], "%Y%m%d")
            all_dates.append(dateim)
        return all_dates

    def test_read_date_reference(self, lst_images):
        all_dates = []
        for im_name in lst_images:
            dateim = im_name.split("_")
            dateim = datetime.strptime(dateim[self.item_date_ref], "%Y%m%d")
            all_dates.append(dateim)
        return all_dates

    def identify_pair_images(self, lst_sensor, lst_ref):
        COLUMN_NAMES = [
            "sensor",
            "mask_sensor",
            "reference",
            "mask_reference",
            "overlap",
        ]
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for name_sensor in lst_sensor:
            print(name_sensor)
            mask_im_tif = name_sensor.split(".tif")
            if (self.sensor_name == "Landsat-8" or self.ref_name == "Landsat-9"):
                mask_im_tif = mask_im_tif[0] + "_mask.tif"
            else:
                #mask_im_tif = mask_im_tif[0] + "_MASK.tif"
                mask_im_tif = mask_im_tif[0] + "_mask.tif"
                 
            if os.path.exists(mask_im_tif):

                # input sensor reference
                dateim = name_sensor.split("_")
                dateim = datetime.strptime(dateim[self.item_date_sensor], "%Y%m%d")
                with rasterio.open(mask_im_tif) as dataim:
                    geomIm = dataim.bounds
                    projIm = dataim.crs

                for name_ref in lst_ref:
                    mask_ref_tif = name_ref.split(".tif")
                    if (self.ref_name == "Sentinel-2" or self.ref_name == "Landsat-8"):
                        mask_ref_tif = mask_ref_tif[0] + "_mask.tif"
                    else:
                        #mask_ref_tif = mask_ref_tif[0] + "_MASK.tif"
                        mask_ref_tif = mask_ref_tif[0] + "_mask.tif"
                                             
                    if os.path.exists(mask_ref_tif):

                        # input raster reference
                        dateref = name_ref.split("_")
                        dateref = datetime.strptime(
                            dateref[self.item_date_ref], "%Y%m%d"
                        )  # important position de la date
                        with rasterio.open(mask_ref_tif) as dataref:
                            geomRef = dataref.bounds
                            projRef = dataref.crs

                        # check date (fmt *_yyyymmdd_RTOC_UINT16_BGRN.tif)
                        delta = abs(dateref - dateim)
                        
                        if (abs(delta.days)) < 6:
                            # check proj
                            if projRef == projIm:

                                # check overlap
                                overlap = self.get_iou(geomRef,geomIm)
                                if overlap > 0:  # si overlap au moins 30%

                                    df_c = pd.DataFrame(
                                        [
                                            [
                                                name_sensor,
                                                mask_im_tif,
                                                name_ref,
                                                mask_ref_tif,
                                                overlap,
                                                abs(delta.days)
                                            ]
                                        ],
                                        columns=[
                                            "sensor",
                                            "mask_sensor",
                                            "reference",
                                            "mask_reference",
                                            "overlap",
                                            "delta_days"
                                        ],
                                    )
                                    df = pd.concat([df, df_c], ignore_index=True)
                                    print(name_ref)                              
        df.set_index([pd.Index(np.arange(0,len(df),1))])
        return df

    def metrics_pair_images(self, name_sensor, name_ref):
        COLUMN_NAMES = [
            "sensor",
            "mask_sensor",
            "reference",
            "mask_reference",
            "overlap",
        ]
        df = pd.DataFrame(columns=COLUMN_NAMES)
        print(name_sensor)
        mask_im_tif = name_sensor.split(".tif")
        if (self.sensor_name == "Landsat-8" or self.ref_name == "Landsat-9"):
            mask_im_tif = mask_im_tif[0] + "_ACM.tif"
        else:
            mask_im_tif = mask_im_tif[0] + "_MASK.tif"
        if os.path.exists(mask_im_tif):

            # input sensor reference
            dateim = name_sensor.split("_")
            dateim = datetime.strptime(dateim[self.item_date_sensor], "%Y%m%d")
            with rasterio.open(mask_im_tif) as dataim:
                geomIm = dataim.bounds
                projIm = dataim.crs
            
            print(name_ref)  
            mask_ref_tif = name_ref.split(".tif")
            if (self.ref_name == "Sentinel-2" or self.ref_name == "Landsat-8"):
                mask_ref_tif = mask_ref_tif[0] + "_ACM.tif"
            else:
                mask_ref_tif = mask_ref_tif[0] + "_MASK.tif"
            if os.path.exists(mask_ref_tif):

                # input raster reference
                dateref = name_ref.split("_")
                dateref = datetime.strptime(
                    dateref[self.item_date_ref], "%Y%m%d"
                )  # important position de la date
                with rasterio.open(mask_ref_tif) as dataref:
                    geomRef = dataref.bounds
                    projRef = dataref.crs

                # check date (fmt *_yyyymmdd_RTOC_UINT16_BGRN.tif)
                delta = abs(dateref - dateim)
                if (abs(delta.days)) < 6:

                    # check proj
                    if projRef == projIm:

                        # check overlap
                        overlap = self.get_iou(geomRef,geomIm)

        return overlap, abs(delta.days)


    def process_pair_image(
        self,
        df,
        im,
        b_size,
        bool_decalib
    ):
        sensor_fileName = df['sensor'].loc[im]
        mask_im_tif = df['mask_sensor'].loc[im]
        ref_fileName = df['reference'].loc[im]
        mask_ref_tif = df['mask_reference'].loc[im]

        with rasterio.open(sensor_fileName) as dataim:
            geomIm = dataim.bounds

        # dataset pre-process
        dp = dataim.res[0] / 2
        xlim1 = geomIm[0] - dp
        ylim1 = geomIm[1] - dp
        xlim2 = geomIm[2] + dp
        ylim2 = geomIm[3] + dp

        # resampling (average) reference image on resolution of sensor image
        ficTmp3 = os.path.join(self.path_tmp, "tmp3" + str(random.randint(1,100000)) + ".tif")
        commandLine = (
            "gdalwarp -q -overwrite -te "
            + str(xlim1)
            + " "
            + str(ylim1)
            + " "
            + str(xlim2)
            + " "
            + str(ylim2)
            + " -ts "
            + str(dataim.width)
            + " "
            + str(dataim.height)
            + " -r average \""
            + ref_fileName
            + "\" \""
            + ficTmp3
            + "\""
        )
        os.system(commandLine)

        # resampling (near) reference image mask on resolution of sensor image
        ficTmp4 = os.path.join(self.path_tmp, "tmp4" + str(random.randint(1,100000)) + ".tif")
        commandLine = (
            "gdalwarp -q -overwrite -te "
            + str(xlim1)
            + " "
            + str(ylim1)
            + " "
            + str(xlim2)
            + " "
            + str(ylim2)
            + " -ts "
            + str(dataim.width)
            + " "
            + str(dataim.height)
            + " -r near \""
            + mask_ref_tif
            + "\" \""
            + ficTmp4
            +"\""
        )
        os.system(commandLine)

        # resampling (near) sensor mask on resolution of sensor image
        ficTmp5 = os.path.join(self.path_tmp, "tmp5" + str(random.randint(1,100000)) + ".tif")
        commandLine = (
            "gdalwarp -q -overwrite -te "
            + str(xlim1)
            + " "
            + str(ylim1)
            + " "
            + str(xlim2)
            + " "
            + str(ylim2)
            + " -ts "
            + str(dataim.width)
            + " "
            + str(dataim.height)
            + " -r near \""
            + mask_im_tif
            + "\" \""
            + ficTmp5
            +"\""
        )
        os.system(commandLine)

        # process mask clear
        with rasterio.open(ficTmp4) as maskref_cur:
            mask_ref = maskref_cur.read(1)
        with rasterio.open(ficTmp5) as maskim_cur:
            mask_im = maskim_cur.read(1)

        mask_ref_reduced = self.resample_mask_by_average(mask_ref, b_size)
        mask_im_reduced = self.resample_mask_by_average(mask_im, b_size)

        # process each band
        for iband in range(0, len(self.band_relat)):
            band_cur = self.band_relat[iband]
            with rasterio.open(sensor_fileName) as dataim_cur:
                band_im = (
                    dataim_cur.read(iband+1) * self.gain_sensor + self.bias_sensor
                )
                if bool_decalib:
                    filename_coeff = os.path.join(
                        self.path_coeff,
                        "Inter_"
                        + self.sensor_name
                        + "_"
                        + self.band_ranking[iband]
                        + ".txt",
                    )
                    lut = self.read_lut(filename_coeff)
                    lut_deca = np.zeros((2, 121))
                    lut_deca[0] = lut[1]
                    lut_deca[1] = lut[0]
                    band_im = self.apply_lut(lut_deca, band_im)
            with rasterio.open(ficTmp3) as dataref_cur:
                if self.ref_name == "Sentinel-2":
                    band_ref = dataref_cur.read(band_cur) * self.gain_ref + self.bias_ref
                else:
                    band_ref = dataref_cur.read(band_cur) * self.gain_ref + self.bias_ref
                    if bool_decalib:
                        filename_coeff = os.path.join(
                            self.path_coeff,
                            "Inter_"
                            + self.ref_name
                            + "_"
                            + self.band_ranking[iband]
                            + ".txt",
                        )
                        lut = self.read_lut(filename_coeff)
                        lut_deca = np.zeros((2, 121))
                        lut_deca[0] = lut[1]
                        lut_deca[1] = lut[0]
                        band_ref = self.apply_lut(lut_deca, band_ref)
                
            band_im_reduced = self.resample_band_by_average(
                band_im, b_size, mask_im_reduced, mask_ref_reduced
            )
            band_ref_reduced = self.resample_band_by_average(
                band_ref, b_size, mask_im_reduced, mask_ref_reduced
            )
            if self.band_ranking[iband]=="Nir":
                band_ref_nir = band_ref_reduced
                band_im_nir = band_im_reduced
            if self.band_ranking[iband]=="Red":
                band_ref_red = band_ref_reduced
                band_im_red = band_im_reduced
            pngfile = sensor_fileName.split("/")
            pngfile = pngfile[-1].split(".tif")
            tmp_png = ref_fileName.split("/")
            tmp_png = tmp_png[-1].split(".tif")
            pngfile = pngfile[-2] + tmp_png[-2] + self.band_ranking[iband] + ".png"
            outfileName = os.path.join(self.outpath_plt,pngfile)

            idx = np.isfinite(band_im_reduced.flatten()) & np.isfinite(
                band_ref_reduced.flatten()
            )
            if np.count_nonzero(idx) > 10:
                df = self.scatter_cal(
                    band_im_reduced.flatten(),
                    band_ref_reduced.flatten(),
                    self.band_ranking[iband],
                    self.sensor_name,
                    self.ref_name,
                    outfileName,
                    df,
                    im
                )
        # scatter plot ndvi
        band_im = (band_im_nir - band_im_red) / (band_im_nir + band_im_red)
        band_ref = (band_ref_nir - band_ref_red) / (band_ref_nir + band_ref_red)

        pngfile = sensor_fileName.split("/")
        pngfile = pngfile[-1].split(".tif")
        tmp_png = ref_fileName.split("/")
        tmp_png = tmp_png[-1].split(".tif")
        pngfile = pngfile[-2] + tmp_png[-2] + "ndvi" + ".png"
        outfileName = os.path.join(self.outpath_plt,pngfile)
        idx = np.isfinite(band_im.flatten()) & np.isfinite(
            band_ref.flatten()
        )
        if np.count_nonzero(idx) > 10:
            df = self.scatter_cal(
                band_im.flatten(),
                band_ref.flatten(),
                "ndvi",
                self.sensor_name,
                self.ref_name,
                outfileName,
                df,
                im
            )
        os.remove(ficTmp3)
        os.remove(ficTmp4)
        os.remove(ficTmp5)
        return df
    
    
    def process_pair_image_WithoutMask(
        self,
        df,
        im,
        b_size,
        bool_decalib
    ):
        sensor_fileName = df['sensor'].loc[im]
        ref_fileName = df['reference'].loc[im]

        with rasterio.open(sensor_fileName) as dataim:
            geomIm = dataim.bounds

        # dataset pre-process
        dp = dataim.res[0] / 2
        xlim1 = geomIm[0] - dp
        ylim1 = geomIm[1] - dp
        xlim2 = geomIm[2] + dp
        ylim2 = geomIm[3] + dp

        # Decomment to check if crosscalibration.py correctly reloaded
        #print("TEST_HCT_1")

        # resampling (average) reference image on resolution of sensor image
        ficTmp3 = os.path.join(self.path_tmp, "tmp3" + str(random.randint(1,100000)) + ".tif")
        commandLine = (
            "gdalwarp -q -overwrite -te "
            + str(xlim1)
            + " "
            + str(ylim1)
            + " "
            + str(xlim2)
            + " "
            + str(ylim2)
            + " -ts "
            + str(dataim.width)
            + " "
            + str(dataim.height)
            + " -r average \""
            + ref_fileName
            + "\" \""
            + ficTmp3
            + "\""
        )

        #Debug
        # commandLine = "gdalinfo "+ref_fileName
        print(f"\n {commandLine}")

        #os.system(commandLine)
        prog = subprocess.Popen(commandLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = prog.communicate()

        # This makes the wait possible
        prog_status = prog.wait()

        # Display Output
        print(f"Error raised from command: {err} \n")

        # To be used for manual set up of reprojected file
        #ficTmp3 = '/Users/aqy/Desktop/JILIN/1_Data/tmp/tmp386036.tif'

        #print('self.gain_ref=',self.gain_ref)
        # process each band
        for iband in range(0, len(self.band_relat)):
            print(f"{datetime.today()} - Process band {iband+1}/{len(self.band_relat)+1}")
            band_cur = self.band_relat[iband]

            # Open tested sensor image
            with rasterio.open(sensor_fileName) as dataim_cur:
                # Modification AQY
                band_im = dataim_cur.read(iband+1)
                band_im = np.where(band_im==0, np.nan, band_im)         # check fill value JILIN 0
                band_im = band_im * self.gain_sensor + self.bias_sensor
                #band_im = (
                #    dataim_cur.read(iband+1) * self.gain_sensor + self.bias_sensor
                #)
                
                if bool_decalib:
                    filename_coeff = os.path.join(
                        self.path_coeff,
                        "Inter_"
                        + self.sensor_name
                        + "_"
                        + self.band_ranking[iband]
                        + ".txt",
                    )
                    lut = self.read_lut(filename_coeff)
                    lut_deca = np.zeros((2, 121))
                    lut_deca[0] = lut[1]
                    lut_deca[1] = lut[0]
                    band_im = self.apply_lut(lut_deca, band_im)

            # Open reference image
            with rasterio.open(ficTmp3) as dataref_cur:
                if self.ref_name == "Sentinel-2":
                    # Modification AQY
                    band_ref = dataref_cur.read(band_cur)
                    band_ref = np.where(band_ref==0, np.nan, band_ref)
                    band_ref = band_ref * self.gain_ref + self.bias_ref
                    #band_ref = dataref_cur.read(band_cur) * self.gain_ref + self.bias_ref
                else:
                    band_ref = dataref_cur.read(band_cur) * self.gain_ref + self.bias_ref
                    if bool_decalib:
                        filename_coeff = os.path.join(
                            self.path_coeff,
                            "Inter_"
                            + self.ref_name
                            + "_"
                            + self.band_ranking[iband]
                            + ".txt",
                        )
                        lut = self.read_lut(filename_coeff)
                        lut_deca = np.zeros((2, 121))
                        lut_deca[0] = lut[1]
                        lut_deca[1] = lut[0]
                        band_ref = self.apply_lut(lut_deca, band_ref)

            # Process
            band_im_reduced = self.resample_band_by_average_WithoutMask(
                band_im, b_size)
            band_ref_reduced = self.resample_band_by_average_WithoutMask(
                band_ref, b_size)
            if self.band_ranking[iband]=="Nir":
                band_ref_nir = band_ref_reduced
                band_im_nir = band_im_reduced
            if self.band_ranking[iband]=="Red":
                band_ref_red = band_ref_reduced
                band_im_red = band_im_reduced

            if os.name == "nt":
                pngfile = sensor_fileName.split("\\")
            else:
                pngfile = sensor_fileName.split("/")

            pngfile = pngfile[-1].split(".tif")

            if os.name == "nt":
                tmp_png = ref_fileName.split("\\")
            else:
                tmp_png = ref_fileName.split("/")

            tmp_png = tmp_png[-1].split(".tif")
            pngfile = pngfile[-2] + tmp_png[-2] + self.band_ranking[iband] + ".png"
            outfileName = os.path.join(self.outpath_plt,pngfile)
            print(f"Output figure ({self.band_ranking[iband]}) file: {outfileName}")

            idx = np.isfinite(band_im_reduced.flatten()) & np.isfinite(
                band_ref_reduced.flatten()
            )

            if np.count_nonzero(idx) > 10:
                df = self.scatter_cal(
                    band_im_reduced.flatten(),
                    band_ref_reduced.flatten(),
                    self.band_ranking[iband],
                    self.sensor_name,
                    self.ref_name,
                    outfileName,
                    df,
                    im
                )
        # scatter plot ndvi
        band_im = (band_im_nir - band_im_red) / (band_im_nir + band_im_red)
        band_ref = (band_ref_nir - band_ref_red) / (band_ref_nir + band_ref_red)

        if os.name == "nt":
            pngfile = sensor_fileName.split("\\")
        else:
            pngfile = sensor_fileName.split("/")

        pngfile = pngfile[-1].split(".tif")

        if os.name == "nt":
            tmp_png = ref_fileName.split("\\")
        else:
            tmp_png = ref_fileName.split("/")

        tmp_png = tmp_png[-1].split(".tif")
        pngfile = pngfile[-2] + tmp_png[-2] + "ndvi" + ".png"
        outfileName = os.path.join(self.outpath_plt,pngfile)

        print(f"Output figure (ndvi) file: {outfileName}")

        idx = np.isfinite(band_im.flatten()) & np.isfinite(
            band_ref.flatten()
        )
        if np.count_nonzero(idx) > 10:
            df = self.scatter_cal(
                band_im.flatten(),
                band_ref.flatten(),
                "ndvi",
                self.sensor_name,
                self.ref_name,
                outfileName,
                df,
                im
            )
        os.remove(ficTmp3)
        print(f"{datetime.today()} - Processing done \n")
        return df
        
    def compute_coeff(
        self,
        df,
        b_size,
        bool_decalib,
    ):
        all_0_ref_reduced =[]
        all_0_im_reduced =[]
        all_1_ref_reduced =[]
        all_1_im_reduced =[]
        all_2_ref_reduced =[]
        all_2_im_reduced =[]
        all_3_ref_reduced =[]
        all_3_im_reduced =[]
        
        for im in range(0,len(df)):
            
            sensor_fileName = df['sensor'].iloc[im]
            mask_im_tif = df['mask_sensor'].iloc[im]
            ref_fileName = df['reference'].iloc[im]
            mask_ref_tif = df['mask_reference'].iloc[im]
            
            print('sensor_fileName=',sensor_fileName)
            
            with rasterio.open(sensor_fileName) as dataim:
                    geomIm = dataim.bounds

            # dataset pre-process
            dp = dataim.res[0] / 2
            xlim1 = geomIm[0] - dp
            ylim1 = geomIm[1] - dp
            xlim2 = geomIm[2] + dp
            ylim2 = geomIm[3] + dp

            # resampling (average) reference image on resolution of sensor image
            ficTmp3 = os.path.join(self.path_tmp, "tmp3" + str(random.randint(1,100000)) + ".tif")
            commandLine = (
                "gdalwarp -q -overwrite -te "
                + str(xlim1)
                + " "
                + str(ylim1)
                + " "
                + str(xlim2)
                + " "
                + str(ylim2)
                + " -ts "
                + str(dataim.width)
                + " "
                + str(dataim.height)
                + " -r average \""
                + ref_fileName
                + "\" \""
                + ficTmp3
                +"\""
            )
            os.system(commandLine)

            # resampling (near) reference image mask on resolution of sensor image
            ficTmp4 = os.path.join(self.path_tmp, "tmp4" + str(random.randint(1,100000)) + ".tif")
            commandLine = (
                "gdalwarp -q -overwrite -te "
                + str(xlim1)
                + " "
                + str(ylim1)
                + " "
                + str(xlim2)
                + " "
                + str(ylim2)
                + " -ts "
                + str(dataim.width)
                + " "
                + str(dataim.height)
                + " -r near \""
                + mask_ref_tif
                + "\" \""
                + ficTmp4
                +"\""
            )
            os.system(commandLine)

            # resampling (near) sensor mask on resolution of sensor image
            ficTmp5 = os.path.join(self.path_tmp, "tmp5" + str(random.randint(1,100000)) + ".tif")
            commandLine = (
                "gdalwarp -q -overwrite -te "
                + str(xlim1)
                + " "
                + str(ylim1)
                + " "
                + str(xlim2)
                + " "
                + str(ylim2)
                + " -ts "
                + str(dataim.width)
                + " "
                + str(dataim.height)
                + " -r near \""
                + mask_im_tif
                + "\" \""
                + ficTmp5
                +"\""
            )
            os.system(commandLine)

            print('step2')
            
            # process mask clear
            with rasterio.open(ficTmp4) as maskref_cur:
                mask_ref = maskref_cur.read(1)
            with rasterio.open(ficTmp5) as maskim_cur:
                mask_im = maskim_cur.read(1)

            mask_ref_reduced = self.resample_mask_by_average(mask_ref, b_size)
            mask_im_reduced = self.resample_mask_by_average(mask_im, b_size)
            
            print('step3')
            # process each band
            for num_band in range(0, len(self.band_relat)):
                print(num_band)
                band_cur = self.band_relat[num_band]
                with rasterio.open(sensor_fileName) as dataim_cur:
                    band_im = (
                        dataim_cur.read(num_band+1) * self.gain_sensor + self.bias_sensor
                    )
                    if bool_decalib:
                        if im==0:
                            print('not apply lut')
                        else:
                            filename_coeff = os.path.join(
                                self.path_coeff,
                                "Inter_"
                                + self.sensor_name
                                + "_"
                                + self.band_ranking[num_band]
                                + ".txt",
                            )
                            lut = self.read_lut(filename_coeff)
                            lut_deca = np.zeros((2, 121))
                            lut_deca[0] = lut[1]
                            lut_deca[1] = lut[0]
                            band_im = self.apply_lut(lut_deca, band_im)
                with rasterio.open(ficTmp3) as dataref_cur:
                    band_ref = dataref_cur.read(band_cur) * self.gain_ref + self.bias_ref

                band_im_reduced = self.resample_band_by_average(
                    band_im, b_size, mask_im_reduced, mask_ref_reduced
                )
                band_ref_reduced = self.resample_band_by_average(
                    band_ref, b_size, mask_im_reduced, mask_ref_reduced
                )

                if num_band==0:
                    all_0_im_reduced.append(band_im_reduced.flatten())
                    all_0_ref_reduced.append(band_ref_reduced.flatten())
                elif num_band==1:
                    all_1_im_reduced.append(band_im_reduced.flatten())
                    all_1_ref_reduced.append(band_ref_reduced.flatten())
                elif num_band==2:
                    all_2_im_reduced.append(band_im_reduced.flatten())
                    all_2_ref_reduced.append(band_ref_reduced.flatten())
                elif num_band==3:
                    all_3_im_reduced.append(band_im_reduced.flatten())
                    all_3_ref_reduced.append(band_ref_reduced.flatten())

        os.remove(ficTmp3)
        os.remove(ficTmp4)
        os.remove(ficTmp5)

        print('exit')
        return all_0_im_reduced,all_1_im_reduced,all_2_im_reduced,all_3_im_reduced, \
            all_0_ref_reduced,all_1_ref_reduced,all_2_ref_reduced,all_3_ref_reduced


    def fill_nan(self, x):
        """
        interpolate to fill nan values"""
        inds = np.arange(x.shape[0])
        good = np.where(np.isfinite(x))
        f = interpolate.interp1d(inds[good], x[good], bounds_error=False)
        y = np.where(np.isfinite(x), x, f(inds))
        return y

    def write_lut(self, gain, offset, outfilename):
        intro_str = (self.sensor_name +" To " + self.ref_name)
        lut = np.zeros((121, 2))
        x = np.linspace(0, 1.2, 121)
        y = np.linspace(0, 1.2, 121)
        y[1:50] = np.minimum(1.2,np.maximum(x[1:50] * gain + offset,0))
        y[50:120] = np.NAN
        y_interp = self.fill_nan(y)
        lut = np.vstack((x, y_interp))
        np.around(lut, decimals=2)
        with open(outfilename, "w") as f:
            f.write("%s\n" % intro_str)
            f.write("%s\n" % "Version S2_2.1")
            for i in range(0, len(lut[0])):
                line = (
                    str(np.around(lut[0][i], decimals=5))
                    + " "
                    + str(np.around(lut[1][i], decimals=5))
                )
                f.write(line)
                f.write("\n")
        return lut

    def read_lut(self, filename):
        X = pd.read_csv(filename, sep="\t", header=None)
        lut = np.zeros((2, 121))
        for i in range(2, len(X)):
            esp_index = X[0][122].index(" ")
            lut[0][i - 2] = float((X[0][i][0:esp_index]))
            lut[1][i - 2] = float((X[0][i][esp_index + 1 :]))
        return lut

    def apply_lut(self, lut, band_im):
        newval = np.zeros(np.shape(band_im))
        for item in range(0, len(lut[0]) - 1):
            val = band_im[
                np.where((band_im > lut[0][item]) & (band_im <= lut[0][item + 1]))
            ]
            gain = (lut[1][item] - lut[1][item + 1]) / (lut[0][item] - lut[0][item + 1])
            offset = (
                lut[0][item] * lut[1][item + 1] - lut[0][item + 1] * lut[1][item]
            ) / (lut[0][item] - lut[0][item + 1])
            newval[
                np.where((band_im > lut[0][item]) & (band_im <= lut[0][item + 1]))
            ] = (gain * val + offset)
        return newval
    
    def classifier_score(self,score,interval0,interval1,interval2):
        if (score >= interval0[0]) & (score < interval0[1]) :
            return 1
        elif (score >= interval1[0]) & (score < interval1[1]) :
            return 2
        elif (score >= interval2[0]) & (score < interval2[1]) :
            return 3
        else:
            return 4
        
    def classify_df(self,df):
        interval0 = [-0.01,0.05]
        interval1 = [-0.03,0.075]
        interval2 = [-0.05,0.1]
        df['classe_Nir_rmse'] = df['Nir_rmse'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0.8,1.2]
        interval1 = [0.5,1.5]
        interval2 = [0.2,1.8]
        df['classe_Nir_gain'] = df['Nir_gain'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.05,0.05]
        interval1 = [-0.1,0.1]
        interval2 = [-0.15,0.15]
        df['classe_Nir_offset'] = df['Nir_offset'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.05,0.05]
        interval1 = [-0.075,0.075]
        interval2 = [-0.1,0.1]
        df['classe_Nir_diffmed'] = df['Nir_diffmed'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0,10]
        interval1 = [10,15]
        interval2 = [15,20]
        df['classe_Nir_pourcoutrange'] = df['Nir_pourcoutrange'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0,10]
        interval1 = [10,15]
        interval2 = [15,20]
        df['classe_Red_rmse'] = df['Red_rmse'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0.8,1.2]
        interval1 = [0.5,1.5]
        interval2 = [0.2,1.8]
        df['classe_Red_gain'] = df['Red_gain'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.02,0.02]
        interval1 = [-0.04,0.04]
        interval2 = [-0.06,0.06]
        df['classe_Red_offset'] = df['Red_offset'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.01,0.02]
        interval1 = [-0.02,0.03]
        interval2 = [-0.03,0.04]
        df['classe_Red_diffmed'] = df['Red_diffmed'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0,10]
        interval1 = [10,15]
        interval2 = [15,20]
        df['classe_Red_pourcoutrange'] = df['Red_pourcoutrange'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.01,0.05]
        interval1 = [-0.03,0.075]
        interval2 = [-0.05,0.1]
        df['classe_Green_rmse'] = df['Green_rmse'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0.8,1.2]
        interval1 = [0.5,1.5]
        interval2 = [0.2,1.8]
        df['classe_Green_gain'] = df['Green_gain'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.02,0.02]
        interval1 = [-0.04,0.04]
        interval2 = [-0.06,0.06]
        df['classe_Green_offset'] = df['Green_offset'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.01,0.02]
        interval1 = [-0.02,0.03]
        interval2 = [-0.03,0.04]
        df['classe_Green_diffmed'] = df['Green_diffmed'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0,10]
        interval1 = [10,15]
        interval2 = [15,20]
        df['classe_Green_pourcoutrange'] = df['Green_pourcoutrange'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.01,0.05]
        interval1 = [-0.03,0.075]
        interval2 = [-0.05,0.1]
        df['classe_ndvi_rmse'] = df['ndvi_rmse'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [0.8,1.2]
        interval1 = [0.65,1.35]
        interval2 = [0.5,1.5]
        df['classe_ndvi_gain'] = df['ndvi_gain'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.1,0.1]
        interval1 = [-0.2,0.2]
        interval2 = [-0.3,0.3]
        df['classe_ndvi_offset'] = df['ndvi_offset'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        interval0 = [-0.05,0.05]
        interval1 = [-0.1,0.1]
        interval2 = [-0.15,0.15]
        df['classe_ndvi_diffmed'] = df['ndvi_diffmed'].apply(self.classifier_score, args=(interval0,interval1,interval2))
        
        interval0 = [0,10]
        interval1 = [10,15]
        interval2 = [15,20]
        df['classe_ndvi_pourcoutrange'] = df['ndvi_pourcoutrange'].apply(self.classifier_score, args=(interval0,interval1,interval2))

        df['Score_Nir'] = (df['classe_Nir_rmse']+df['classe_Nir_gain']+df['classe_Nir_offset']+df['classe_Nir_diffmed']+df['classe_Nir_pourcoutrange'])/5
        df['Score_Red'] = (df['classe_Red_rmse']+df['classe_Red_gain']+df['classe_Red_offset']+df['classe_Red_diffmed']+df['classe_Red_pourcoutrange'])/5
        df['Score_Green'] = (df['classe_Green_rmse']+df['classe_Green_gain']+df['classe_Green_offset']+df['classe_Green_diffmed']+df['classe_Green_pourcoutrange'])/5
        df['Score_ndvi'] = (df['classe_ndvi_rmse']+df['classe_ndvi_gain']+df['classe_ndvi_offset']+df['classe_ndvi_diffmed']+df['classe_ndvi_pourcoutrange'])/5

        df['Score'] = (df['Score_Nir']+df['Score_Red']+df['Score_Green']+df['Score_ndvi'])/4

        df.drop(['overlap','delta_days','Nir_r2','Nir_std','Nir_rmse','Nir_gain','Nir_diffmed','Nir_diffm', 'Nir_pourcoutrange','Nir_offset','Red_offset', 'Red_gain',
       'Red_diffm', 'Red_rmse', 'Red_std', 'Red_pourcoutrange', 'Red_r2',
       'Red_diffmed', 'Blue_pourcoutrange', 'Blue_std', 'Blue_rmse',
       'Blue_gain', 'Blue_diffmed', 'Blue_diffm', 'Blue_r2', 'Blue_offset',
       'Green_pourcoutrange', 'Green_rmse', 'Green_std', 'Green_diffmed',
       'Green_diffm', 'Green_gain', 'Green_offset', 'Green_r2', 'ndvi_diffmed',
       'ndvi_std', 'ndvi_diffm', 'ndvi_gain',
#       'ndvi_std', 'ndvi_diffm', 'ndvi_r2', 'ndvi_gain', 'ndvi_pourcoutrange',
       'ndvi_rmse', 'ndvi_offset'], axis=1, inplace=True) 

        return df