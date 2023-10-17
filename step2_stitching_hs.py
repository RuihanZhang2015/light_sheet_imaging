import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
# from PIL import Image


def compute_mean_intensity(
        layer_path,
        period = 30,
        first_frames = 300):
    '''
    Compute mean intensity of the first few frames of each z layer.
    Input: 
        layer_path: a string with {} as the placeholder for camera index.
    Output: 
        mean_intensity: a 2D array of shape (period, first_frames).
    '''

    # mean_intensity = np.zeros((period, first_frames))
    # with h5py.File(layer_path, 'r') as f:
    #     for layer_index in range(period):
    #         vol = f[f'layer{layer_index}'][:, :, :first_frames]
    #         mean_intensity[layer_index,:] = np.mean(vol,axis=(0,1),keepdims=False) 
    # assert mean_intensity.shape == (period, first_frames)
    # return mean_intensity

    mean_intensity = np.zeros((period, first_frames))
    for layer_index in range(period):
        with h5py.File(layer_path.format(layer_index), 'r') as f:
            vol = f[f'layer{layer_index}']
            vol = vol[:, :, :first_frames]
            mean_intensity[layer_index,:] = np.mean(vol,axis=(0,1),keepdims=False)
        print('computing average intensity of layer', layer_index)
    assert mean_intensity.shape == (period, first_frames)
    return mean_intensity

def compute_laser_onset_per_layer(mean_intensity, period = 30):
    '''
    Compute laser onset for each layer.
    Input: 
        mean_intensity, a 2D array of shape (period, first_frames).
    Output: 
        A list of length 30, each element is the onset time for that layer.
    '''

    result = [0] * period
    for layer_index in range(period):
        this_layer = mean_intensity[layer_index,:]
        result[layer_index] = np.argmax(np.diff(this_layer))+1
    return result


def find_laser_onset(onset_time_list):
    '''
    Find the ealiest onset for the time.
    Input: 
        onset_time_list, a list of length 30, each element is the onset time for that layer.
    Output: 
        onset_layer (int), onset_time (int).
    '''
    # If the onset time already follows the order(from small to large), return the first one.
    is_sorted = np.all(onset_time_list == np.sort(onset_time_list))
    if is_sorted:
        return 0, onset_time_list[0]
    
    # If the first few layers have later onset time, return the first smaller one.
    onset_layer = np.argmin(np.diff(onset_time_list)) + 1
    assert onset_time_list[onset_layer] == min(onset_time_list)
    return onset_layer, onset_time_list[onset_layer]


def stitching(
    cam1_path,
    cam2_path,
    out_path,
    cam1_onset_layer,
    cam1_onset_time,
    cam2_onset_layer,
    cam2_onset_time,
    customized_function,
    offset_time,
    desired_len
    ):
    '''
    Input:
        cam1_path, cam2_path: path to the h5 files of the two cameras, they have datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, t).
        out_path: path to the output h5 file, which will have datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, t).
        cam1_onset_layer, cam1_onset_time: the layer and time of the first laser onset in cam1.
        cam2_onset_layer, cam2_onset_time: the layer and time of the first laser onset in cam2.
        customized_function: a function that takes two images and return a stitched image.
        offset_time: we only consider time after cam1 onset time + offset_time.
        desired_len: the desired number of frames in the output volume.

    Output:
        A h5 file with datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, desired_len).

    Explanation:
        # Layer x in Camera1 , Layer y in camera2
        # Final layer_z, t0 = concat(camera1[layer_x, t1], camera2[layer_y, t2])
        # Example:
        # Camera1: Onset_time 143, Onset_layer 10
        # Camera2: Onset_time 196, Onset_layer 20
        # Mapping: layer 10 <-> 20, 11 <-> 21, 12 <-> 22, 13 <-> 23, 14 <-> 24, 15 <-> 25, 16 <-> 26, 17 <-> 27, 18 <-> 28, 
        # 19 <-> 29, 20 <-> 0, 21 <-> 1, 22 <-> 2, 23 <-> 3, 24 <-> 4, 25 <-> 5, 26 <-> 6, 27 <-> 7, 28 <-> 8, 29 <-> 9
        # Time: 143 <-> 196 ....
        # Let's say t0 = t1 + 100
        # Stiched Layer 0:  Camera1 (Layer 10, t=243) + Camera2(Layer 20, t=296)
        # Stiched Layer 1:  Camera1 (Layer 11, t=244) + Camera2(Layer 21, t=297)

    Variables:
        t0: starting time globally (unit is according to the first camera)
        t1: actually the same as the t0, for clarity.
        t2: starting time in the second camera.
        layer_order: layers with smaller onset frame index will be placed in the front.
        layer_final: the layer index in the final volume.
        layer_cam1: the layer index in camera1.
        layer_cam2: the layer index in camera2.
    '''
    
    t0 = cam1_onset_time + offset_time
    layer_order = np.roll(range(30), cam1_onset_layer)

    for layer_final,layer_cam1 in enumerate(layer_order):
        print(f'Stitching {layer_final}/{len(layer_order)}')

        # Computer corresponding layers in cam2
        layer_cam2 = (layer_cam1+cam2_onset_layer-cam1_onset_layer) % 30

        # Compute corresponding time in cam2
        t1 = t0
        t2 = t0+cam2_onset_time-cam1_onset_time

        # Load data from cam1 and cam2
        with h5py.File(cam1_path.format(layer_cam1),'r') as f:
            vol1 = f[f'layer{layer_cam1}']
            vol1 = vol1[:,:,t1:t1+desired_len]
            assert vol1.shape[2] == desired_len, f'desired number of frames is {desired_len}, but the actual number of frames is {vol1.shape[2]}'

        with h5py.File(cam2_path.format(layer_cam2),'r') as f:
            vol2 = f[f'layer{layer_cam2}']
            vol2 = vol2[:,:,t2:t2+desired_len]
            assert vol2.shape[2] == desired_len, f'desired number of frames is {desired_len}, but the actual number of frames is {vol2.shape[2]}'
        
        # Stitch two volumes
        vol = []
        for i in range(desired_len):
            im1 = vol1[:,:,i]
            im2 = vol2[:,:,i]
            stitched = customized_function(im1,im2)
            vol.append(stitched)
        vol = np.stack(vol, axis=0)
        vol = np.transpose(vol, (1,2,0))
        
        with h5py.File(out_path.format(layer_final),'w') as f:
            dataset_name, data_shape, data_type = f'layer{layer_final}', vol.shape, vol.dtype 
            if dataset_name in f:
                del f[dataset_name]
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
            dataset[:] = vol
            
    print('Final vol shape',vol.shape)

def plot_mean_intensity(mean_intensity, camera_index, save_dir = '/om2/user/zgwang/zeguan/image'):
    '''
    Plot mean intensity and diff(mean_intensity) for each layer.
    
    Input: 
        mean_intensity: a 2D array of shape (period, first_frames).
        camera_index: 1 or 2.
        save_dir: folder to save the plots.
    '''

    plt.figure(figsize = (20,10))
    for layer_index in range(30):
        plt.plot(mean_intensity[layer_index,:],'.-',label = f'layer{layer_index}')
    plt.xlabel('frame')
    plt.ylabel('mean_intensity')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'mean_intensity_cam{camera_index}.jpg'))
    plt.close()

    plt.figure(figsize = (20,10))
    for layer_index in range(30):
        plt.plot(np.diff(mean_intensity[layer_index,:]),'.-',label = f'layer{layer_index}')
        plt.xlabel('frame')
    plt.ylabel('diff(mean_intensity)')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'diff mean_intensity_cam{camera_index}.jpg'))
    plt.close()


def plot_stitched(out_path, layer_index = 0, save_dir = '/om2/user/zgwang/zeguan/image'):
    '''
    Plot the stitched image of a layer.
    Input:
        out_path: path to the output h5 file, which will have datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, t).
        layer_index: index of the layer.
    '''
    
    with h5py.File(out_path,'r') as f:
        img = f[f'layer{layer_index}'][:,:,25]
    plt.figure()
    plt.imshow(img,vmax = 200)
    plt.savefig(os.path.join(save_dir,f'stitched_{layer_index}.jpg'))
    plt.close()

def combining_two_cameras(layer_path_cam1, layer_path_cam2, stitch_path, stitch_fn, desired_len, offset_time):
    '''
    Input:
        layer_path_cam1/2: path to the volume from each camera, a string with {} as the placeholder for camera index. It has datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, t).
        stitch_path: path to the output h5 file, which will have datasets as f[f'layer{layer_index}'], with the shape of (256, 1280, t).
        desired_len: the desired number of frames in the output volume.
        offset_time: we only consider time after cam1 onset time + offset_time.
    '''

    # Compute mean intensity of the first few frames of each z layer
    mean_intensity_cam1 = compute_mean_intensity(layer_path_cam1)
    mean_intensity_cam2 = compute_mean_intensity(layer_path_cam2)
    
    # plot_mean_intensity(
    #     mean_intensity = mean_intensity_cam1,
    #     camera_index = 1,
    #     save_dir = os.path.dirname(stitch_path))

    # plot_mean_intensity(
    #     mean_intensity = mean_intensity_cam2,
    #     camera_index = 2,
    #     save_dir = os.path.dirname(stitch_path))

    # Compute laser onset for each layer
    cam1_onset_per_layer = compute_laser_onset_per_layer(mean_intensity_cam1)
    cam2_onset_per_layer = compute_laser_onset_per_layer(mean_intensity_cam2)

    print('cam1_onset_per_layer ---------------')
    for i,x in enumerate(cam1_onset_per_layer):
        print(i,x)
    print('cam2_onset_per_layer ---------------')
    for i,x in enumerate(cam2_onset_per_layer):
        print(i,x)
    
    # Find the layer with the earliest laser onset
    cam1_onset_layer, cam1_onset_time = find_laser_onset(cam1_onset_per_layer)
    cam2_onset_layer, cam2_onset_time = find_laser_onset(cam2_onset_per_layer)

    print('---------------------------')
    print(f'cam1_onset_layer {cam1_onset_layer} cam1_onset_time {cam1_onset_time}')
    print(f'cam2_onset_layer {cam2_onset_layer} cam2_onset_time {cam2_onset_time}')

    stitch_folder_path = os.path.dirname(stitch_path)
    print('stitch_folder_path', stitch_folder_path)
    os.makedirs(stitch_folder_path, exist_ok=True)

    print('---------------------------')
    stitching(
        cam1_path = layer_path_cam1,
        cam2_path = layer_path_cam2,
        out_path = stitch_path,
        cam1_onset_layer = cam1_onset_layer,
        cam1_onset_time = cam1_onset_time,
        cam2_onset_layer = cam2_onset_layer,
        cam2_onset_time = cam2_onset_time,
        customized_function = stitch_fn,
        offset_time = offset_time,
        desired_len = desired_len
    )

    
if __name__ == '__main__':
    
    # Define paths
    layer_path_cam1 = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/layer_{}.h5'
    layer_path_cam2 = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/layer_{}.h5'
    stitch_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/stitched/layer_{}.h5'
    from custom_function.stitch2cam_0826fish5 import stitch2cam_0826fish5 as stitch_fn

    # TEST
    combining_two_cameras(layer_path_cam1, layer_path_cam2, stitch_path, stitch_fn, desired_len = 50, offset_time = 50) 

    # RUN
    # combining_two_cameras(layer_path,stitch_path,desired_len = 6700, offset_time = 100) 

    # EXAMINE
    # for layer_index in range(30):
    #     plot_stitched(stitch_path,layer_index, save_dir = '/om2/user/zgwang/zeguan/image')
