import numpy as np
import xml.etree.ElementTree as ET
import os
import nrrd
import h5py
import tqdm
import matplotlib.pyplot as plt

def retrieve_framenumbers(meta_data_path):

    '''
    Takes in metafaile file path and returns a list of frame numbers
    eg. 
    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230820_Gal4_3xPosi2_5dpf_40us_4980us_UV/camera1/fish1_1.xiseq'
    frame_numers = retrieve_framenumbers(meta_data_path, video_initial_drops)
    '''

    tree = ET.parse(meta_data_path) 
    root = tree.getroot()
    frameNumbers = []
    for file_elem in root.findall(".//file"):
        frame = int(file_elem.get("frame"))
        frameNumbers.append(frame)
    
    return frameNumbers 


def nrrd_to_h5(input_nrrd_path, chunk_index, all_h5_path):
    '''
    Takes in nrrd file path and chunk_h5_path and saves the nrrd file as a chunk_h5_path
    eg. 
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish1/fish1_1_{}.nrrd'
    chunk_h5_path = '/nese/mit/group/boydenlab/ruihan/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish1/fish1_1_{}.h5'
    for chunk_index in range(1,8):
        nrrd_to_h5(input_nrrd_path.format(chunk_index), chunk_h5_path.format(chunk_index))
    '''

    print(f'processing {chunk_index}')
    h5_folder_path = os.path.dirname(all_h5_path)
    os.makedirs(h5_folder_path, exist_ok=True)

    nrrd_data,_ = nrrd.read(input_nrrd_path)
    nrrd_data = np.transpose(nrrd_data, (1, 0, 2))
    print(nrrd_data.shape)

    with h5py.File(all_h5_path,'a') as f:
        dataset_name = f'{chunk_index}'  # Name of the dataset in HDF5
        data_shape = nrrd_data.shape  # Shape of the NRRD data
        data_type = nrrd_data.dtype  # Data type of the NRRD data
        if dataset_name in f:
            dataset = f[dataset_name]
        else:
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
        dataset[:] = nrrd_data


# def stitch_chunks(
#         video_initial_drops,
#         all_h5_path, 
#         height = 256,
#         width = 1280,
#         chunk_size = 10000,
#         num_chunks = 7):
#     '''
#     Stitch several chunks(nrrd files) along the last axis as an entire volume(h5 file). Then crop the volume to remove the initial few volumes (i.e. volumes * 30 slices).
#     '''
    
#     data = np.zeros((height, width, num_chunks * chunk_size), dtype=np.uint16)
#     for chunk_index in range(1,num_chunks+1):
#         print(f'processing {chunk_index}')
#         with h5py.File(all_h5_path,'r') as f:
#             chunk_data = f.get(f'{chunk_index}')[:]
#             data[:, :, (chunk_index-1) * chunk_size:chunk_index * chunk_size] = chunk_data

#     with h5py.File(all_h5_path,'a') as f:
#         dataset_name, data_shape, data_type = 'all',  data.shape , data.dtype 
#         if dataset_name in f:
#             dataset = f[dataset_name]
#         else:
#             dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
#         dataset[:] = data[:,:,video_initial_drops * _PERIOD : ]
# # stitch_chunks(video_initial_drops, all_h5_path)


def assign_layer(frame_numbers, video_initial_drops):
    '''
    Takes in a list of frame numbers and returns a dictionary of layer and corresponding frame numbers.
    For example: 
        result = {
            layer0: [[0,0],[30,30],[60,60],        [119,120],[149,150]] --> [0,30,60,60,119,149]
            layer1: [[1,1],[31,31],[61,61],[90,91],[120,121],[150,151]] --> [1,31,61,90,120,150]
        }
    '''
    import collections
    result = collections.defaultdict(list)
    for i, frame_number in enumerate(frame_numbers):
        if i < video_initial_drops * _PERIOD:
            continue
        layer = frame_number % _PERIOD
        result[layer].append(tuple([i,frame_number]))

    final = {}
    for layer in range(30):
        entry = result[layer]
        final[layer] = [entry[0][0]]
        for i in range(1, len(entry)):
            if 60 > entry[i][1] - entry[i-1][1] > 30:
                final[layer].append(entry[i-1][0])
            elif entry[i][1] - entry[i-1][1] > 60:
                final[layer].extend([entry[i-1][0]]*2)
            final[layer].append(entry[i][0])
    return final

def process_layer(
        layer_index, 
        frame_indexes_per_layer, 
        all_h5_path, 
        layer_h5_path,
        debug = False,
        height = 256,
        width = 1280):
    '''
    Put together the slices that belong to layer_index.
    frame_indexes_per_layer: frame indexes for this layer.
    Input path: all_h5_path
    Output path: layer_h5_path.
    '''
    frame_indexes = frame_indexes_per_layer[layer_index]
    if debug:
        frame_indexes = frame_indexes[:1000]
    print(len(frame_indexes))
    layer = np.zeros((height, width, len(frame_indexes)), dtype=np.uint16)
    with h5py.File(all_h5_path,'r') as f:
        for i, frame_index in tqdm.tqdm(enumerate(frame_indexes)):
            layer[:,:,i] = f.get(f'{frame_index//10000+1}')[:,:,frame_index%10000]
    
    with h5py.File(layer_h5_path,'a') as f:
        dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
        if dataset_name in f:
            dataset = f[dataset_name]
        else:
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
        dataset[:] = layer

def check_layer(layer_index):
    
    plt.figure()
    with h5py.File(layer_h5_path,'r') as f:
        vol = f[f'layer{layer_index}'][:,:,:1100]
    vol = np.mean(vol,axis=(0,1),keepdims=False)
    print(vol.shape)
    plt.plot(vol,'.-')
    plt.savefig('/om2/user/ruihanz/zeguan/layer{layer_index}.png')

# def check_volume(x,y):
#     plt.figure()
#     vol = []
#     for layer_index in range(_PERIOD):
#         with h5py.File(layer_h5_path,'r') as f:
#             vol.append(f[f'layer{layer_index}'][:,x,y])
#     vol = np.array(vol)
#     plt.imshow(vol,vmax = 300)
#     plt.savefig('/om2/user/ruihanz/zeguan/t50.png')

_PERIOD = 30

# Parse metadata
meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230820_Gal4_3xPosi2_5dpf_40us_4980us_UV/camera1/fish1_1.xiseq'
video_initial_drops = 30
frame_numers = retrieve_framenumbers(meta_data_path)
frame_indexes_per_layer = assign_layer(frame_numers, video_initial_drops)

# Get the entire volume
input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish1/fish1_1_{}.nrrd'
all_h5_path = '/nese/mit/group/boydenlab/ruihan/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera1/fish1_1_raw.h5'
layer_h5_path = '/nese/mit/group/boydenlab/ruihan/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera1/fish1_1.h5'

# # Convert nrrd to h5
# for chunk_index in range(1,21):
#     nrrd_to_h5(input_nrrd_path.format(chunk_index), chunk_index, all_h5_path)

# Reorganize the layers
for layer_index in [0,10]:#range(_PERIOD):
    process_layer(layer_index, frame_indexes_per_layer, all_h5_path, layer_h5_path, debug = False)


# layer_index = 10
# check_layer(layer_index)

# x = 300
# y = 600
# check_volume(x,y)

    