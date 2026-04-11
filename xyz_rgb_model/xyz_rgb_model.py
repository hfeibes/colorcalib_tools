"""
Follow color calibration process
Use CSC measurements as control points
"""
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import calib_control_funcs as ccf
except:
    os.chdir('xyz_rgb_model')
    import calib_control_funcs as ccf


tablet = 'amoled_1'
data_dir = 'input_files'
start_with_start_M = True
start_with_start_gamma = True
predict_new = False

# Choose training and testing sets
train_by = 'id' # type or id
# If train_by = id, train and test must be lists of id numbers in the df, otherwise, color types
train = list(range(17)) # ['csc_l', 'csc_d', 'csc_lg', 'csc_dg', 'csc_g', 'monochr', 'gray']
test = list(range(17,65))
control_point_ids = None #list(range(0,15))


# Load all XYZ RGB data
data = pd.read_csv(os.path.join(data_dir, tablet + '_measured_XYZ.csv'))
# expects df with columns color id, r,g,b,x,y,z,type (gray, monochromatic, csc_light)
data = ccf.average_repeated_measurements(data)

if start_with_start_M: # Initialize matrix as calibration code would have
    Minv, start_M = ccf.get_starting_M(data)
else:
    start_M = None
if start_with_start_gamma:
    # Get starting gamma params from forward fit of calib data
    calib_r,calib_g,calib_b,calib_w = ccf.split_calib_df(data, drop0=True)
    start_gamma_params = []
    for i, channel in enumerate([calib_r,calib_g,calib_b]):
        xdat = channel[['r', 'g', 'b']].to_numpy()
        xdat = xdat[:,i]
        xdat = xdat/255 # normalize to max 
        ydat = channel['y'].to_numpy()
        ydat = ydat/np.max(ydat, axis=0) # normalize to max in each channel
        pa, pb, pg = ccf.get_starting_gamma(.01,1,2.2,xdat,ydat)
        start_gamma_params.append([pa,pb,pg])
    start_gamma = [start_gamma_params[0][2],start_gamma_params[1][2],start_gamma_params[2][2]]
else: start_gamma = None
    

def train_calib_model(epochs, model, loss_fn, param_optim, train_xyz, train_rgb, 
                      test_xyz=None, test_rgb=None, test=False, loss_fn2=None, 
                      loss_weights=None, control_point_idxs = None):
    train_loss_ = []
    epoch_i = []
    for epoch in range(epochs):
        model.train()
        rgb_pred = model(train_xyz)
        loss = loss_fn(rgb_pred, train_rgb)
        if loss_fn2 is not None:
            loss2 = loss_fn2(rgb_pred, train_rgb)
            loss2 = loss2[control_point_idxs]
            loss = loss.mean()*loss_weights[0] + loss2.mean()*loss_weights[1]
            #print('loss ', loss.mean(), ' loss 2 ', loss2.mean())
        else:
            loss = loss.mean()

        param_optim.zero_grad()
        loss.backward()
        param_optim.step()
        
        if torch.isnan(loss):
            print('Nan in loss at epoch ', epoch)
            for name, param in calib_mod.named_parameters():
                if torch.isnan(param).any():
                    print('nan in param ', name)
                if torch.isinf(param).any():
                    print('inf in param ', name)
                    
        model.eval()
        with torch.inference_mode():
            train_rgb_pred = model(train_xyz)
            
            if test:
                test_rgb_pred = model(test_xyz)
                test_loss = loss_fn(test_rgb_pred, test_rgb)
            else:
                test_rgb_pred = None
            
            if epoch % 10 == 0:
                print(epoch, loss)
                if test:
                    print(test_loss)                
                train_loss_.append(loss.detach().numpy())
                epoch_i.append(epoch)
    return train_rgb_pred, test_rgb_pred, train_loss_



# Set up model
calib_mod = ccf.CalibControl(start_M=start_M, start_gamma=start_gamma)
# Define loss
loss_fn = nn.MSELoss(reduction='none')
param_optim = torch.optim.Adam(params=calib_mod.parameters(), lr=1e-3)
epochs = 18000

# Compound with L1 loss for control points
loss_fn2 = nn.L1Loss(reduction='none')
# Add ability for shuffling or something, but this works well rn
if train:
    train_df = data[data[train_by].isin(train)]
    train_xyz = train_df[['x', 'y', 'z']].to_numpy()
    train_rgb = train_df[['r', 'g', 'b']].to_numpy()
    train_rgb = ccf.normalize_rgb(train_rgb)
    train_xyz = torch.from_numpy(train_xyz).to(torch.float32)
    train_rgb = torch.from_numpy(train_rgb).to(torch.float32)
    
    if control_point_ids is not None:
        control_point_idxs = train_df.index[train_df['id'].isin(control_point_ids)].tolist()
    else:
        control_point_idxs=None

if test:
    test_df = data[data[train_by].isin(test)] 
    test_xyz = test_df[['x', 'y', 'z']].to_numpy()
    test_rgb = test_df[['r', 'g', 'b']].to_numpy()
    test_rgb = ccf.normalize_rgb(test_rgb)
    test_xyz = torch.from_numpy(test_xyz).to(torch.float32)
    test_rgb = torch.from_numpy(test_rgb).to(torch.float32)
else:
    test_xyz=None
    test_rgb=None

train_rgb_pred, test_rgb_pred, train_loss_ = train_calib_model(epochs, 
                                                               calib_mod, 
                                                               loss_fn, 
                                                               param_optim, 
                                                               train_xyz, 
                                                               train_rgb, 
                                                               test_xyz=test_xyz, 
                                                               test_rgb=test_rgb,
                                                               test=test,
                                                               loss_fn2=loss_fn2,
                                                               loss_weights=[.5,.5],
                                                               control_point_idxs=control_point_idxs) #.85,.15 or .5,.5

fig,axs= ccf.plot_colors(train_rgb, train_rgb_pred)
plt.show()
plt.close()
if test:
    fig,axs= ccf.plot_colors(test_rgb, test_rgb_pred)
    plt.show()
    plt.close()

plt.plot(train_loss_)
plt.show()
plt.close()

#torch.save(calib_mod.state_dict(), os.path.join(data_dir, tablet, 'calib_model.pth'))

if predict_new:
    xyztopredict = pd.read_csv('/mnt/isilon/PROJECTS/ColorCategoriesTablet/color_definitions/atlantis/atlantis_CC_LUVXYZ_redlum65_try100.csv', 
                               header=None, names=['L','u','v', 'x', 'y', 'z'])
    new_xyz = xyztopredict[['x', 'y', 'z']].to_numpy()
    new_xyz = torch.from_numpy(new_xyz).to(torch.float32)
    predict_new_rgb = ccf.CalibControl(start_M, start_gamma)
    predict_new_rgb.load_state_dict(torch.load(os.path.join(data_dir, tablet, 'calib_model.pth'), weights_only=True))
    predict_new_rgb.eval()
    with torch.inference_mode():
        new_rgb = predict_new_rgb(new_xyz)
        new_rgb = new_rgb.detach().numpy()
    
    clip_new_rgb = np.clip(new_rgb,0,1)
    pred_rgb_df = pd.DataFrame(np.floor(new_rgb*255), columns = ['r','g','b'])
    full_tablet_info = pd.concat([xyztopredict,pred_rgb_df], axis=1)
    #full_tablet_info['color_id'] = full_tablet_info['color_id']-1 #reset to 0-indexing, bc of matlab interfacing changing it
    
    n_colors = len(full_tablet_info)
    # Get 64 circle xy coord
    polar = np.linspace(0, 360, n_colors+1)
    radians = np.deg2rad(polar)
    radius = 48 # get from another read in
    x = radius * np.cos(radians)
    y = radius * np.sin(radians)
    fig, axs = plt.subplots()
    #axs.set_facecolor(clip_new_rgb[-1])
    axs.scatter(x[0:n_colors],y[0:n_colors],c=clip_new_rgb)#[0:7])
    axs.set_aspect('equal')
    plt.show()
    plt.close()
    #full_tablet_info.to_csv('/mnt/isilon/PROJECTS/ColorCategoriesTablet/color_definitions/enterprise/enterprise_CC_RGB_redlum65.csv', index=False)
