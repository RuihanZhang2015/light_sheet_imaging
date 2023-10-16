import numpy as np
import xml.etree.ElementTree as ET
import os
import nrrd
import h5py
import tqdm
import matplotlib.pyplot as plt

_PERIOD = 30

def retrieve_framenumbers(meta_data_path):
    '''
    Takes in metafaile file path and returns a list of frame numbers
    Input:
        meta_data_path: path to the metafile.
    Output:
        frameNumbers: a list of frame numbers.
    '''

    tree = ET.parse(meta_data_path) 
    root = tree.getroot()
    frameNumbers = []
    for file_elem in root.findall(".//file"):
        frame = int(file_elem.get("frame"))
        frameNumbers.append(frame)
    return frameNumbers 


def nrrd_to_h5(input_nrrd_path, chunk_index, raw_path):
    '''
    Convert nrrd files to h5 files.
    Input:
        input_nrrd_path: path to the nrrd file.
        chunk_index: index of the chunk.
        raw_path: path to the output h5 file.
    Output:
        A h5 file with 21 datasets, one for each chunk, named as f['{chunk_index}']
    '''

    print(f'processing {chunk_index}')
    h5_folder_path = os.path.dirname(raw_path)
    print('h5_folder_path', h5_folder_path)
    os.makedirs(h5_folder_path, exist_ok=True)

    nrrd_data,_ = nrrd.read(input_nrrd_path)
    nrrd_data = np.transpose(nrrd_data, (1, 0, 2))

    with h5py.File(raw_path,'a') as f:
        dataset_name = f'{chunk_index}'  # Name of the dataset in HDF5
        data_shape = nrrd_data.shape  # Shape of the NRRD data
        data_type = nrrd_data.dtype  # Data type of the NRRD data
        if dataset_name in f:
            dataset = f[dataset_name]
        else:
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
        dataset[:] = nrrd_data


def assign_layer(frame_numbers, video_initial_drops):
    '''
    Takes in a list of frame numbers and returns a dictionary of layer and corresponding frame numbers.
    For example: 
        result = {
            layer0: [[0,0],[30,30],[60,60],        [119,120],[149,150]] --> [0,30,60,60,119,149]
            layer1: [[1,1],[31,31],[61,61],[90,91],[120,121],[150,151]] --> [1,31,61,90,120,150]
        }
    Input:
        frame_numbers: a list of frame numbers.
        video_initial_drops: number of frames to drop at the beginning of the video.
    Output:
        result: a dictionary of layer and corresponding frame numbers.
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
            difference = entry[i][1] - entry[i-1][1]
            final[layer].extend([entry[i-1][0]]* (difference//30-1))
            final[layer].append(entry[i][0])

    return final


# def process_layer(
#         layer_index, 
#         frame_indexes_per_layer, 
#         raw_path, 
#         layer_path,
#         debug = False,
#         height = 256,
#         width = 1280):
#     '''
#     Put together the slices that belong to each layer.
#     Input:
#         layer_index: index of the layer.
#         frame_indexes_per_layer: a dictionary of layer and corresponding frame numbers.
#         raw_path: path to the input h5 file, with 21 datasets, one for each chunk, named as f['{chunk_index}']
#         layer_path: path to the output h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
#         debug: if True, only process the first 300 frames.
#         height: height of the image.
#         width: width of the image.
#     Output:
#         A h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
#     '''
#     frame_indexes = frame_indexes_per_layer[layer_index]
#     if debug:
#         frame_indexes = frame_indexes[:300]
#     print(len(frame_indexes))
#     layer = np.zeros((height, width, len(frame_indexes)), dtype=np.uint16)
#     with h5py.File(raw_path,'r') as f:
#         for i, frame_index in enumerate(frame_indexes):
#             layer[:,:,i] = f.get(f'{frame_index//10000+1}')[:,:,frame_index%10000]
    
#     if os.path.exists(layer_path):
#         with h5py.File(layer_path,'a') as f:
#             dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
#             if dataset_name in f:
#                 del f[dataset_name]
#             dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
#             dataset[:] = layer
#     else:
#         with h5py.File(layer_path,'w') as f:
#             dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
#             if dataset_name in f:
#                 del f[dataset_name]
#             dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
#             dataset[:] = layer


def check_layer(layer_path, layer_index, save_dir = '/om2/user/zgwang/light_sheet_imaging/image'):
    '''
    Plot the mean intensity of each layer to see if it is continuous to knwo if the layer is processed correctly.
    Input:
        layer_path: path to the output h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
        layer_index: index of the layer.
        save_dir: directory to save the plot.
    Output:
        A plot of the mean intensity of this layer.
    '''
    plt.figure()
    with h5py.File(layer_path,'r') as f:
        vol = f[f'layer{layer_index}'][:,:,:]
    vol = np.mean(vol,axis=(0,1),keepdims=False)
    plt.plot(vol,'.-')
    plt.savefig(os.path.join(save_dir,f'layer_{layer_index}.png'))
    plt.close()


# def check_volume(x,y):
#     plt.figure()
#     vol = []
#     for layer_index in range(_PERIOD):
#         with h5py.File(layer_path,'r') as f:
#             vol.append(f[f'layer{layer_index}'][:,x,y])
#     vol = np.array(vol)
#     plt.imshow(vol,vmax = 300)
#     plt.savefig('/om2/user/zgwangz/zeguan/t50.png')


# def process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, video_initial_drops = 30, debug = False):
#     '''
#     Process all the data from one camera.
#     Input:
#         meta_data_path: path to the metafile.
#         input_nrrd_path: path to the nrrd file.
#         raw_path: path to the output h5 file with 21 datasets, one for each chunk, named as f['{chunk_index}']
#         layer_path: path to the output h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
#     Output: 
#         A h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
#     '''

#     frame_numers = retrieve_framenumbers(meta_data_path)
#     frame_indexes_per_layer = assign_layer(frame_numers, video_initial_drops)
#     frame_len = len(frame_numers)

#     # # Convert nrrd to h5
#     # for chunk_index in range(1,22):
#     #     nrrd_to_h5(input_nrrd_path.format(chunk_index), chunk_index, raw_path)

#     height = 256
#     width = 1280
#     raw_frames = np.zeros((height, width, frame_len), dtype=np.uint16)

#     with h5py.File(raw_path,'r') as f:
#         for i in range(frame_len):
#             raw_frames[:,:,i] = f.get(f'{i//10000+1}')[:,:,i%10000]
#     print('finish reading all raw frames')

#     # Reorganize the layers
#     for layer_index in range(_PERIOD):
#         print('processing layer', layer_index)
#         # process_layer(layer_index, frame_indexes_per_layer, raw_path, layer_path.format(layer_index), debug = debug)
        
#         frame_indexes = frame_indexes_per_layer[layer_index]

#         if debug:
#             frame_indexes = frame_indexes[:300]
#         print(len(frame_indexes))
#         layer = np.zeros((height, width, len(frame_indexes)), dtype=np.uint16)

#         for i, frame_index in enumerate(frame_indexes):
#             layer[:,:,i] = raw_frames[:,:,frame_index]

#         with h5py.File(layer_path.fourmat(layer_index),'w') as f:
#             dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
#             if dataset_name in f:
#                 del f[dataset_name]
#             dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
#             dataset[:] = layer

def process_one_camera(meta_data_path, input_nrrd_path, raw_path, layer_path, video_initial_drops=30, debug=False):
    frame_numbers = retrieve_framenumbers(meta_data_path)
    frame_indexes_per_layer = assign_layer(frame_numbers, video_initial_drops)
    frame_len = len(frame_numbers)

    height = 256
    width = 1280
    raw_frames = np.zeros((height, width, frame_len), dtype=np.uint16)

    # read all raw frames from nrrd chunks
    total_chunks = 21
    chunk_size = 10000

    for chunk_index in range(1, total_chunks + 1):
        nrrd_data, _ = nrrd.read(input_nrrd_path.format(chunk_index))
        
        # Transpose the data
        nrrd_data = np.transpose(nrrd_data, (1, 0, 2))
        
        # Calculate start and end positions for the chunk in the raw_frames
        start_idx = (chunk_index - 1) * chunk_size
        end_idx = chunk_index * chunk_size

        # Insert the data into raw_frames
        raw_frames[:, :, start_idx:end_idx] = nrrd_data

        print(f'Read chunk {chunk_index}/{total_chunks}')


    # # Read raw frames from h5
    # with h5py.File(raw_path, 'r') as f:
    #     for i in range(frame_len):
    #         raw_frames[:, :, i] = f.get(f'{i//10000 + 1}')[:, :, i % 10000]
    #         if i % 1 == 0:
    #             print('reading...', i)

    print('Finished reading all raw frames')

    # Reorganize the layers
    for layer_index in range(_PERIOD):
        print('Processing layer', layer_index)
        
        frame_indexes = frame_indexes_per_layer[layer_index]
        
        if debug:
            frame_indexes = frame_indexes[:300]

        # Get slices of the raw frames corresponding to the frame_indexes
        layer_frames = raw_frames[:, :, frame_indexes]

        with h5py.File(layer_path.format(layer_index), 'w') as f:
            dataset_name = f'layer{layer_index}'
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=layer_frames)


if __name__ == '__main__':

    # meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/xiseq files/fish1_1.xiseq'
    # input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish1/fish1_1_{}.nrrd'
    # raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1_1/camera1/raw.h5'
    # layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1_1/camera1/layer.h5'
    # # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True) 
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False) 
    
    # meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/xiseq files/fish1_1.xiseq'
    # input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/nrrd/fish1/fish1_1_{}.nrrd'
    # raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1_1/camera2/raw.h5'
    # layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1_1/camera2/layer.h5'
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True) 
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False) 

    # # 0826 fish3_1 camera1
    # # nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3
    # meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/xiseq files/fish3_1.xiseq'
    # input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3/fish3_1_{}.nrrd'
    # raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/raw.h5'
    # layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/layer.h5'
    # # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True)
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False)

    # # 0826 fish3_1 camera2
    # meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/xiseq files/fish3_1.xiseq'
    # input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/nrrd/fish3/fish3_1_{}.nrrd'
    # raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/raw.h5'
    # layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/layer.h5'
    # # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True)
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False)

    # 0826 fish3_1 camera1, save to 30 hdf5 files
    # nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3
    # meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/xiseq files/fish3_1.xiseq'
    # input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3/fish3_1_{}.nrrd'
    # raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/raw.h5'
    # layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera1/layer_{}.h5'
    # # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True)
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False)

    # 0826 fish3_1 camera2, save to 30 hdf5 files
    # nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish3
    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/xiseq files/fish3_1.xiseq'
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/nrrd/fish3/fish3_1_{}.nrrd'
    raw_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/raw.h5'
    layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish3_1/camera2/layer_{}.h5'
    # process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = True)
    process_one_camera(meta_data_path,input_nrrd_path,raw_path,layer_path, debug = False)
    
    