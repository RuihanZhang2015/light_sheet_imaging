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


def nrrd_to_h5(input_nrrd_path, chunk_index, all_path):
    '''
    Convert nrrd files to h5 files.
    Input:
        input_nrrd_path: path to the nrrd file.
        chunk_index: index of the chunk.
        all_path: path to the output h5 file.
    Output:
        A h5 file with 21 datasets, one for each chunk, named as f['{chunk_index}']
    '''

    print(f'processing {chunk_index}')
    h5_folder_path = os.path.dirname(all_path)
    print('h5_folder_path', h5_folder_path)
    os.makedirs(h5_folder_path, exist_ok=True)

    nrrd_data,_ = nrrd.read(input_nrrd_path)
    nrrd_data = np.transpose(nrrd_data, (1, 0, 2))

    with h5py.File(all_path,'a') as f:
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


def process_layer(
        layer_index, 
        frame_indexes_per_layer, 
        all_path, 
        layer_path,
        debug = False,
        height = 256,
        width = 1280):
    '''
    Put together the slices that belong to each layer.
    Input:
        layer_index: index of the layer.
        frame_indexes_per_layer: a dictionary of layer and corresponding frame numbers.
        all_path: path to the input h5 file, with 21 datasets, one for each chunk, named as f['{chunk_index}']
        layer_path: path to the output h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
        debug: if True, only process the first 300 frames.
        height: height of the image.
        width: width of the image.
    Output:
        A h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
    '''
    frame_indexes = frame_indexes_per_layer[layer_index]
    if debug:
        frame_indexes = frame_indexes[:300]
    print(len(frame_indexes))
    layer = np.zeros((height, width, len(frame_indexes)), dtype=np.uint16)
    with h5py.File(all_path,'r') as f:
        for i, frame_index in enumerate(frame_indexes):
            layer[:,:,i] = f.get(f'{frame_index//10000+1}')[:,:,frame_index%10000]
    
    if os.path.exists(layer_path):
        with h5py.File(layer_path,'a') as f:
            dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
            if dataset_name in f:
                del f[dataset_name]
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
            dataset[:] = layer
    else:
        with h5py.File(layer_path,'w') as f:
            dataset_name, data_shape, data_type = f'layer{layer_index}', layer.shape , layer.dtype 
            if dataset_name in f:
                del f[dataset_name]
            dataset = f.create_dataset(dataset_name, data_shape, dtype=data_type)
            dataset[:] = layer


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


def process_one_camera(meta_data_path,input_nrrd_path,all_path,layer_path, video_initial_drops = 30, debug = False):
    '''
    Process all the data from one camera.
    Input:
        meta_data_path: path to the metafile.
        input_nrrd_path: path to the nrrd file.
        all_path: path to the output h5 file with 21 datasets, one for each chunk, named as f['{chunk_index}']
        layer_path: path to the output h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
    Output: 
        A h5 file with 30 datasets, one for each layer, named as f['layer{layer_index}']
    '''

    frame_numers = retrieve_framenumbers(meta_data_path)
    frame_indexes_per_layer = assign_layer(frame_numers, video_initial_drops)

    # Convert nrrd to h5
    for chunk_index in range(1,22):
        nrrd_to_h5(input_nrrd_path.format(chunk_index), chunk_index, all_path)

    # Reorganize the layers
    for layer_index in range(_PERIOD):
        print('processing layer', layer_index)
        process_layer(layer_index, frame_indexes_per_layer, all_path, layer_path, debug = debug)


if __name__ == '__main__':

    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/xiseq files/fish1_1.xiseq'
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera1/nrrd/fish1/fish1_1_{}.nrrd'
    all_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera1/fish1_1_raw.h5'
    layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera1/fish1_1.h5'
    # process_one_camera(meta_data_path,input_nrrd_path,all_path,layer_path, debug = True) 
    process_one_camera(meta_data_path,input_nrrd_path,all_path,layer_path, debug = False) 
    
    meta_data_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/xiseq files/fish1_1.xiseq'
    input_nrrd_path = '/nese/mit/group/boydenlab/symvou/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/camera2/nrrd/fish1/fish1_1_{}.nrrd'
    all_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera2/fish1_1_raw.h5'
    layer_path = '/nese/mit/group/boydenlab/zgwang/FISHDATA/VOLTAGE/20230826_gal4_3xPosi2_xCaspr_F2_5-6dpf_40us_4980us_UV/fish1/camera2/fish1_1.h5'
    # process_one_camera(meta_data_path,input_nrrd_path,all_path,layer_path, debug = True) 
    process_one_camera(meta_data_path,input_nrrd_path,all_path,layer_path, debug = False) 


    
    