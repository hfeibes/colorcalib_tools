import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor, sqrt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up model
class CalibControl(nn.Module):
    # Define neural net
    def __init__(self,start_M,start_gamma):
        super(CalibControl, self).__init__()
        self.start_M  = start_M
        self.M = nn.Parameter(torch.tensor(self.start_M,dtype=torch.float32), requires_grad=True) #nn.Parameter(torch.eye(3) + .01, requires_grad=True)
        self.start_gamma = start_gamma #torch.ones(3) * -.5
        self.gamma = nn.Parameter(torch.tensor(self.start_gamma,dtype=torch.float32), requires_grad=True) # second layer, mimic gamma correction 
        self.nonlinfix = nn.Softplus()
        self.start_bias = torch.zeros(3) + .01
        self.bias = nn.Parameter(torch.tensor(self.start_bias,dtype=torch.float32), requires_grad=True)
        
# =============================================================================
#         self.gamma_net = nn.Sequential(
#             nn.Linear(3, 4),  #12 # 6 for 15 examples
#             nn.ReLU(),
#             nn.Linear(4, 3), # output a gamma per channel
#             nn.Softplus()) # ensure gamma>0 , maybe not needed
# =============================================================================

    # Define pass through model
    def forward(self, xyz):
        self.rgb_pre_gamma = xyz
        lin_rgb = xyz @ self.M + self.bias#F.softplus(self.bias)
        gamma = F.softplus(self.gamma) + .01
        lin_rgb = lin_rgb.to(torch.complex64)
        rgb = (lin_rgb**(gamma)).real
        return rgb

        

# Functions
def split_calib_df(calib_df, drop0 = False):
    calib_r = calib_df.loc[(calib_df['g']==0)&(calib_df['b']==0)]
    calib_g = calib_df.loc[(calib_df['r']==0)&(calib_df['b']==0)]
    calib_b = calib_df.loc[(calib_df['r']==0)&(calib_df['g']==0)]
    calib_w = calib_df.loc[(calib_df['g']==calib_df['b'])&(calib_df['g']==calib_df['r'])]
    if drop0:
        calib_r = calib_r.loc[calib_r['r']!=0]
        calib_g = calib_g.loc[calib_g['g']!=0]
        calib_b = calib_b.loc[calib_b['b']!=0]
    return calib_r,calib_g,calib_b,calib_w

def get_starting_M(calib_df):
    # matlab equivalent
    rgb2xyY_M = np.zeros((3,3))
    XYZr = calib_df.loc[(calib_df['r']==255)&(calib_df['g']==0)&(calib_df['b']==0)] # get XYZ of max red gun
    XYZg = calib_df.loc[(calib_df['r']==0)&(calib_df['g']==255)&(calib_df['b']==0)]
    XYZb = calib_df.loc[(calib_df['r']==0)&(calib_df['g']==0)&(calib_df['b']==255)]
    XYZw = calib_df.loc[(calib_df['r']==255)&(calib_df['g']==255)&(calib_df['b']==255)]

    for i, v in enumerate([XYZr,XYZg,XYZb]):
        XYZv = [v['x'].values[0],v['y'].values[0],v['z'].values[0]]
        xv = XYZv[0]/np.sum(XYZv) # chroma x
        yv = XYZv[1]/np.sum(XYZv) # chroma y
        Yv = 1-xv-yv
        rgb2xyY_M[i][:] = np.array([xv,yv,Yv])
        
    rgb2xyY_M = rgb2xyY_M.T # each column should cover all three r g b guns

    XYZwhite = np.array([XYZw['x'].values[0],XYZw['y'].values[0],XYZw['z'].values[0]])
    XYZwhite = XYZwhite*683 # candela scaled
    S = np.dot(np.linalg.inv(rgb2xyY_M), XYZwhite)
    
    xyY2XYZ = np.array([[S[0],0,0],[0,S[1],0],[0,0,S[2]]])
    RGB2XYZ = np.dot(rgb2xyY_M, xyY2XYZ)
    
    XYZ2RGB = np.linalg.inv(RGB2XYZ)
    return RGB2XYZ, XYZ2RGB

def normalize_rgb(rgb):
    # Normalize to maximum input gun value (max intensity)
    rgb_ = np.floor(rgb)
    rgb_ = rgb_/255
    return rgb_

def normalize_xyz(xyz, minmax = True):
    if minmax: # get to [0,1] and keep luminance information
        all_max = np.max(xyz)
        all_min = np.min(xyz)
        scaled_xyz = (xyz - all_min) / (all_max - all_min)
    return scaled_xyz

def average_repeated_measurements(df):
    df_= df.groupby(['id']).mean().reset_index()
    return df_

def get_starting_gamma(a,b,g,xdat,ydat):
    def func(x,a,b,g):
        return a + b*x**g
    p0 = [a,b,g]
    params, covariance = curve_fit(func, xdat, ydat, p0=p0)
    pa, pb, pg = params
    return pa, pb, pg


def plot_colors(true, pred, csc=False):
    try:
        plot_rgb = true.detach().numpy()
        plot_pred_rgb = pred.detach().numpy()
    except:
        plot_rgb = true
        plot_pred_rgb = pred
    #plot_pred_rgb = np.clip(plot_pred_rgb, 0, 1) # if something has landed out of gamut, bring it back down for drawing purposes
    
    n = len(plot_rgb)
   
    if csc:
        titles = ['train']*(n-1) + ['test']
        fig, axs = plt.subplots(3, 5, figsize=(20,20))
    fig, axs = plt.subplots(floor(sqrt(n)), floor(sqrt(n))+3, figsize=(20,20))
    axs = axs.flatten()
    for j in range(n):
        pred_label = 'pred'
        for (c, channel) in zip(range(2), ['R','G','B']):
            if plot_pred_rgb[j][c] > 1.:
                pred_label = pred_label+'_'+channel+'+'
            if plot_pred_rgb[j][c] < 0.:
                pred_label = pred_label+'_'+channel+'-'
        plot_pred_rgb[j] = np.clip(plot_pred_rgb[j], 0, 1)
        axs[j].pie([181,181], labels=['true',pred_label],colors=[plot_rgb[j], plot_pred_rgb[j]])
        if csc: 
            axs[j].set_title(titles[j])
    return fig, axs
