"""
Convert spectra measurements from a PR into XYZ coordinates using chosen color matching function (Judd or 1931)
Basically just a python translation of Hansen's matlab script spectra2XYZ.m
Todo: add judd option
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Change these
photodata_path = 'spec_outputs/testout.csv' # csv with spectra to convert
cmf = 'cie1931' # judd or cie1931 
outname = 'testvalues' # name for output csv
outdir = 'spec_outputs' # dir for output csv

def commondomain(x1,y1,x2,y2):
    # interpolate every 1 nm
    if max(np.diff(x1)) > 1:
        x1i = np.arange(x1[0], x1[-1]+1)
        spl = CubicSpline(x1, y1)
        y1i = spl(x1i)
    if max(np.diff(x2)) > 1:
        x2i = np.arange(x2[0], x2[-1]+1)
        spl = CubicSpline(x2, y2)
        y2i = spl(x2i)
    # Find commmon range and domain
    xc = np.arange(max(x1i[0], x2i[0]), min(x1i[-1], x2i[-1]))
    y1c = y1i[np.isin(x1i,xc),:]
    y2c = y2i[np.isin(x2i,xc)]
    return xc, y1c, y2c

# Load color matching data
if cmf == 'cie1931':
    cmf_df = pd.read_csv('color_matching_functions/ciexyzj.txt',  header=None, names=['wavelength', 'x_bar', 'y_bar', 'z_bar'])
x1 = cmf_df['wavelength'].to_numpy()
y1 = cmf_df[['x_bar', 'y_bar', 'z_bar']].to_numpy()

# Load measurements
photodata = pd.read_csv(photodata_path)
measurements = photodata['rep'].unique() # how many repeats of each color
colors = photodata['id'].unique() # how many colors

# Iterate through all measurements
all_xyz = []
for rep in measurements:
    rep_data = photodata[photodata['rep']==rep]
    for color_id in colors:
        color_data = rep_data[rep_data['id']==color_id]
        wavelength_spectra = color_data['nm'].to_numpy()
        spectra_measured = color_data['power'].to_numpy()
        wavelength_common, xyz_common, spectra_measured_common = commondomain(x1,y1,
                                                                              wavelength_spectra,
                                                                              spectra_measured)
        
        XYZ = spectra_measured_common @ xyz_common # units = radiance
        all_xyz.append([rep, color_id, color_data['r'].iloc[0],
                        color_data['g'].iloc[0],color_data['b'].iloc[0],
                        XYZ[0], XYZ[1], XYZ[2]])
measured_xyz = pd.DataFrame(all_xyz, columns = ['rep', 'id', 'r', 'g', 'b', 'X', 'Y', 'Z'])
measured_xyz.to_csv(os.path.join(outdir, outname + '_measured_XYZ.csv'), index=False)
