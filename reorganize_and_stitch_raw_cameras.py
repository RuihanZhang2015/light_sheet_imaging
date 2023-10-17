import os
import nrrd
import h5py
import xml.etree.ElementTree as ET
import numpy as np

PERIOD = 30

################ reorganize raw frames into layers ################################

def retrieve_framenumbers(meta_data_path):
    """Retrieve a list of frame numbers from a metafile.
    
    Args:
        meta_data_path (str): Path to the metafile.

    Returns:
        list: List of frame numbers.
    """
    tree = ET.parse(meta_data_path) 
    root = tree.getroot()
    return [int(file_elem.get("frame")) for file_elem in root.findall(".//file")]

# def nrrd_to_h5(input_nrrd_path, chunk_index, raw_path):
#     """Convert nrrd files to h5 files.
    
#     Args:
#         input_nrrd_path (str): Path to the nrrd file.
#         chunk_index (int): Index of the chunk.
#         raw_path (str): Path to the output h5 file.
#     """
#     print(f'Processing {chunk_index}')
#     os.makedirs(os.path.dirname(raw_path), exist_ok=True)

#     nrrd_data, _ = nrrd.read(input_nrrd_path)
#     nrrd_data = np.transpose(nrrd_data, (1, 0, 2))

#     with h5py.File(raw_path, 'a') as f:
#         dataset_name = str(chunk_index)
#         if dataset_name not in f:
#             f.create_dataset(dataset_name, data=nrrd_data)

def assign_layer(frame_numbers, video_initial_drops):
    """Assign layers to frames and group them.
    
    Args:
        frame_numbers (list): List of frame numbers.
        video_initial_drops (int): Number of frames to drop at the beginning of the video.

    Returns:
        dict: Dictionary of layer and corresponding frame numbers.
    """
    from collections import defaultdict

    result = defaultdict(list)
    for i, frame_number in enumerate(frame_numbers):
        if i < video_initial_drops * PERIOD:
            continue
        layer = frame_number % PERIOD
        result[layer].append((i, frame_number))

    final = {}
    for layer in range(PERIOD):
        entry = result[layer]
        final[layer] = [entry[0][0]]
        for i in range(1, len(entry)):
            difference = entry[i][1] - entry[i-1][1]
            final[layer].extend([entry[i-1][0]] * (difference//PERIOD-1))
            final[layer].append(entry[i][0])

    return final

def process_one_camera(meta_data_path, input_nrrd_path, layer_path, video_initial_drops=30, debug=False):
    """Process one camera's data.
    
    Args:
        meta_data_path (str): Path to the meta data.
        input_nrrd_path (str): Path to the input nrrd files.
        layer_path (str): Path to the layer file.
        video_initial_drops (int): Number of frames to drop at the beginning. Default is 30.
        debug (bool): If in debug mode, only consider 300 frames. Default is False.
    """
    frame_numbers = retrieve_framenumbers(meta_data_path)
    frame_indexes_per_layer = assign_layer(frame_numbers, video_initial_drops)

    height, width, frame_len = 256, 1280, len(frame_numbers)
    raw_frames = np.zeros((height, width, frame_len), dtype=np.uint16)

    os.makedirs(os.path.dirname(layer_path), exist_ok=True)

    total_chunks, chunk_size = 21, 10000

    for chunk_index in range(1, total_chunks + 1):
        nrrd_data, _ = nrrd.read(input_nrrd_path.format(chunk_index))
        nrrd_data = np.transpose(nrrd_data, (1, 0, 2))
        start_idx, end_idx = (chunk_index - 1) * chunk_size, chunk_index * chunk_size
        raw_frames[:, :, start_idx:end_idx] = nrrd_data
        print(f'Read chunk {chunk_index}/{total_chunks}')

    print('Finished reading all raw frames')

    num_frames = len(frame_indexes_per_layer[0])
    if debug:
        num_frames = 300

    first_frames = 300
    avg_intensities_array = np.zeros((PERIOD, first_frames))

    for layer_index in range(PERIOD):
        print(f'Reorganize layer {layer_index}')
        
        frame_indexes = frame_indexes_per_layer[layer_index]
        if debug:
            frame_indexes = frame_indexes[:300]

        layer_frames = raw_frames[:, :, frame_indexes]
        avg_intensities = np.mean(layer_frames, axis=(0, 1),keepdims=False)
        avg_intensities_array[layer_index, :first_frames] = avg_intensities[:first_frames]

        with h5py.File(layer_path.format(layer_index), 'w') as f:
            dataset_name = f'layer{layer_index}'
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=layer_frames)

    with h5py.File(f'{os.path.dirname(layer_path)}/average_intensities.h5', 'w') as f:
        if 'avg_intensities' in f:
            del f['avg_intensities']
        f.create_dataset('avg_intensities', data=avg_intensities_array)
    
    return raw_frames, frame_indexes_per_layer, avg_intensities_array


############################ stitch two cameras ########################################

def compute_laser_onset_per_layer(mean_intensity, period=30):
    '''... (same docstring) ...'''
    return [np.argmax(np.diff(mean_intensity[layer_index, :])) + 1 for layer_index in range(period)]

def find_laser_onset(onset_time_list):
    '''... (same docstring) ...'''
    is_sorted = np.all(onset_time_list == np.sort(onset_time_list))
    if is_sorted:
        return 0, onset_time_list[0]
    onset_layer = np.argmin(np.diff(onset_time_list)) + 1
    return onset_layer, onset_time_list[onset_layer]

def combine_two_volumes(vol1, vol2, stitch_function, desired_len):
    '''Combine two volumes using the provided stitch function.'''
    combined_vol = []
    for i in range(desired_len):
        stitched = stitch_function(vol1[:, :, i], vol2[:, :, i])
        combined_vol.append(stitched)
    return np.transpose(np.stack(combined_vol, axis=0), (1, 2, 0))

def save_combined_volume(out_path, vol, layer_final):
    '''Save the combined volume to the output path.'''
    with h5py.File(out_path.format(layer_final), 'w') as f:
        dataset_name = f'layer{layer_final}'
        if dataset_name in f:
            del f[dataset_name]
        f.create_dataset(dataset_name, data=vol)

def stitching(raw_cam1, frame_indexes_per_layer_cam1,
              raw_cam2, frame_indexes_per_layer_cam2,
              out_path,
              cam1_onset_layer, cam1_onset_time,
              cam2_onset_layer, cam2_onset_time,
              stitch_function,
              offset_time, desired_len):
    '''... (same docstring) ...'''
    t0 = cam1_onset_time + offset_time
    layer_order = np.roll(range(PERIOD), cam1_onset_layer)

    for layer_final, layer_cam1 in enumerate(layer_order):
        print(f'Stitching {layer_final}/{len(layer_order)}')

        layer_cam2 = (layer_cam1 + cam2_onset_layer - cam1_onset_layer) % 30
        t1, t2 = t0, t0 + cam2_onset_time - cam1_onset_time

        frame_indexes_cam1 = frame_indexes_per_layer_cam1[layer_cam1]
        frame_indexes_cam2 = frame_indexes_per_layer_cam2[layer_cam2]

        # layer_frames = raw_frames[:, :, frame_indexes]
        vol1 = raw_cam1[:,:,frame_indexes_cam1]
        vol2 = raw_cam2[:,:,frame_indexes_cam2]

        vol1 = vol1[:, :, t1:t1+desired_len]
        vol2 = vol2[:, :, t2:t2+desired_len]

        combined_vol = combine_two_volumes(vol1, vol2, stitch_function, desired_len)
        save_combined_volume(out_path, combined_vol, layer_final)

########################################################################################

if __name__ == '__main__':

    #### reorganize ####
    # 0826 fish3_1 camera1, save to 30 hdf5 files
    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/xiseq files/fish3_1.xiseq'
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3/fish3_1_{}.nrrd'
    layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/layer_{}.h5'
    raw_frames_1, frame_indexes_per_layer_1, avg_intensities_array_1 = process_one_camera(meta_data_path,input_nrrd_path,layer_path, debug = False)

    # 0826 fish3_1 camera2, save to 30 hdf5 files
    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/xiseq files/fish3_1.xiseq'
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/nrrd/fish3/fish3_1_{}.nrrd'
    layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/layer_{}.h5'
    raw_frames_2, frame_indexes_per_layer_2, avg_intensities_array_2 = process_one_camera(meta_data_path,input_nrrd_path,layer_path, debug = False)
    
    ##### stitch ####
    stitch_out_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/stitched/layer_{}.h5'
    
    os.makedirs(os.path.dirname(stitch_out_path), exist_ok=True)

    from custom_function.stitch2cam_0826fish5 import stitch2cam_0826fish5 as stitch_fn
    onset_time_list_1 = compute_laser_onset_per_layer(avg_intensities_array_1)
    onset_time_list_2 = compute_laser_onset_per_layer(avg_intensities_array_2)
    cam1_onset_layer, cam1_onset_time = find_laser_onset(onset_time_list_1)
    cam2_onset_layer, cam2_onset_time = find_laser_onset(onset_time_list_2)
    stitching(raw_frames_1, frame_indexes_per_layer_1,
              raw_frames_2, frame_indexes_per_layer_2, 
              stitch_out_path,
              cam1_onset_layer, cam1_onset_time, 
              cam2_onset_layer, cam2_onset_time,
              stitch_fn,
              offset_time=100, desired_len=6700)