import os
import numpy as np
import h5py
from customized_stitching_function import customized_function

def compute_mean_intensity(
        layer_h5_path,
        period = 30,
        first_frames = 300):
    '''
    Compute mean intensity of the first few frames of each z layer
    '''
    mean_intensity = np.zeros((period, first_frames))
    with h5py.File(layer_h5_path,'a') as f:
        for layer_index in range(period):
            vol = f[f'layer{layer_index}'][:, :, :first_frames]
        mean_intensity[layer_index,:] = np.mean(vol,axis=(0,1),keepdims=False) 
    return mean_intensity

def compute_laser_onset_per_layer(mean_intensity):
    '''
    '''
    result = {}
    for layer_index in range(30):
        result[layer_index] = np.argmax(np.diff(mean_intensity))+1
    return result

def find_laser_onset(onset_time_list):
    '''
    Find the ealiest onset for the time.
    '''
    indices_of_min = np.where(onset_time_list == np.min(onset_time_list))
    onset_layer = indices_of_min[0][0]
    return onset_time_list[onset_layer], onset_layer # 143, 10

def stitching(
    cam1_path,
    cam2_path,
    out_path,
    cam1_onset_layer,
    cam1_onset_time,
    cam2_onset_layer,
    cam2_onset_time,
    customized_function,
    offset_from_layer = 100,
    desired_len = 6700
    ):
    '''
    # Layer x in Camera1 , Layer y in camera2
    # What's known? 
    # Camera1: Onset_time 143, Onset_layer 10
    # Camera2: Onset_time 196, Onset_layer 20
    # Mapping: layer 10 <-> 20, 11 <-> 21, 12 <-> 22, 13 <-> 23, 14 <-> 24, 15 <-> 25, 16 <-> 26, 17 <-> 27, 18 <-> 28, 19 <-> 29, 20 <-> 0, 21 <-> 1, 22 <-> 2, 23 <-> 3, 24 <-> 4, 25 <-> 5, 26 <-> 6, 27 <-> 7, 28 <-> 8, 29 <-> 9
    # Time: 143 <-> 196 
    # Final layer_z, t0 = concat(camera1[layer_x, t1], camera2[layer_y, t2])
    # Let's say t0 = t1 + 100
    # Stiched Layer 0:  Camera1 (Layer 10, t=243) + Camera2(Layer 20, t=296)
    # Stiched Layer 1:  Camera1 (Layer 11, t=244) + Camera2(Layer 21, t=297)
    '''
        
    t0 = cam1_onset_time + offset_from_layer
    layer_order = np.roll(range(30), cam1_onset_layer)
    for layer_final,layer_cam1 in enumerate(layer_order):

        # Computer corresponding layers in cam2
        layer_cam2 = (layer_cam1+cam2_onset_layer-cam1_onset_layer)%30
        t1 = t0
        t2 = t0+cam2_onset_time-cam1_onset_time

        # Load data from cam1 and cam2
        with h5py.File(cam1_path,'r') as f:
            vol1 = f[f'layer{layer_cam1}'][:,:,t1:t1+desired_len]
        with h5py.File(cam2_path,'r') as f:
            vol2 = f[f'layer{layer_cam2}'][:,:,t2:t2+desired_len]
        
        # Stitch two volumes
        vol = np.stack([customized_function(im1,im2) for im1,im2 in zip(vol1,vol2)],axis=2)
        with h5py.File(out_path,'a') as f:
            dataset_name, data_shape, data_type = f'layer{layer_final}', vol.shape, vol.dtype 
            if dataset_name in f:
                dataset = f[dataset_name]
            else:
                dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
            dataset[:] = vol

def combining_two_cameras(stitch_path,layer_h5_path ):

    # Compute mean intensity of the first few frames of each z layer
    mean_intensity_cam1 = compute_mean_intensity(layer_h5_path.format(1))
    mean_intensity_cam2 = compute_mean_intensity(layer_h5_path.format(2))

    # Compute laser onset for each layer
    cam1_onset_per_layer = compute_laser_onset_per_layer(mean_intensity_cam1)
    cam2_onset_per_layer = compute_laser_onset_per_layer(mean_intensity_cam2)

    # Find the layer with the earliest laser onset
    cam1_onset_layer,cam1_onset_time = find_laser_onset(cam1_onset_per_layer)
    cam2_onset_layer,cam2_onset_time = find_laser_onset(cam2_onset_per_layer)

    stitching(
        cam1_path = layer_h5_path.format(1),
        cam2_path = layer_h5_path.format(2),
        out_path = stitch_path,
        cam1_onset_layer = cam1_onset_layer,
        cam1_onset_time = cam1_onset_time,
        cam2_onset_layer = cam2_onset_layer,
        cam2_onset_time = cam2_onset_time,
        customized_function = customized_function,
        offset_from_layer = 100,
        desired_len = 6700
    )

# Define paths
stitch_path = '/nese/mit/group/boydenlab/ruihan/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/fish1_1_stitched.h5'
layer_h5_path = '/nese/mit/group/boydenlab/ruihan/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera{}/fish1_1.h5'
combining_two_cameras(stitch_path,layer_h5_path)