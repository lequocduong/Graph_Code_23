# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:01:13 2021

@author: zwg
"""
import xmltodict

def get_band_order_GS2(xml_path, expected_band_names):
    """
  xml_path: path to .gil GS2 metadata
  expected_band_names: a list of band names. Should be one of ['red', 'green', 'blue', 'nir'] as of 2020
  returns: a list of band index, variable as read from metadata
  """
    allowed_bands = ['red', 'green', 'blue', 'nir']
    for bn in expected_band_names:
        assert bn.lower() in allowed_bands, "band name should be one of {}, got {}".format(allowed_bands, bn.lower())
    with open(xml_path) as fh:
        xml = fh.read()
    xmldict = xmltodict.parse(xml)
    band_names = {el['DESCRIPTION'].lower():el['INDEX'] for el in xmldict['GEOSYSIMAGEINFO']['BANDS']}
    return [int(el) for el in list((band_names[bn.lower()] for bn in expected_band_names))]