clear all;
​
%load cam1 
cam1_path = 'D:/zebrafish/230820/Fish2_1/camera1/aligned_layers';
cam2_path = 'D:/zebrafish/230820/Fish2_1/camera2/aligned_layers';
stitch_path = 'D:/zebrafish/230820/Fish2_1/stitched/aligned_layers';
​
%load variables 
load('./cam1_file_post.mat')
load('./cam1_rise_time_loc.mat')
load('./cam2_file_post.mat')
load('./cam2_rise_time_loc.mat')
​
desired_len = 6700; 
offset_from_laser = 100; 
start_time = min(cam1_rise_time_loc, cam2_rise_time_loc)+offset_from_laser;
frame_diff = cam2_rise_time_loc - cam1_rise_time_loc;
​
max_row = 505; %
max_col = 1280;
​
stitched_video = zeros(max_col,max_row,desired_len,'uint16');
​
for i=1:30
​
[cam1_file_post(i) cam2_file_post(i)]
​
cam1  = load([cam1_path '/layer_' int2str(cam1_file_post(i)) '.mat']);
cam2 = load([cam2_path '/layer_' int2str(cam2_file_post(i)) '.mat']);
​
    for k = 0:desired_len-1
        stitched_video(:,:,k+1) = stitch2cam_20230820fish2_1(cam1.layer_all_uint(:,:,start_time + k), ...
                                                          cam2.layer_all_uint(:,:,start_time + k + frame_diff))';
    end 
    [start_time + k , start_time + k + frame_diff]
​
    disp(['writing layer: ' int2str(i-1)]);
​
    stitch_hdf_file = [stitch_path '/stitch_layer_' int2str(i-1) '.hdf5'];
    delete(stitch_hdf_file) %delete duplicates 
    h5create(stitch_hdf_file,'/mov',[max_col max_row desired_len],'Datatype','uint16'); %length(sRegions.PixelIdxList)])
    h5write(stitch_hdf_file,'/mov',stitched_video)
​
    h5writeatt(stitch_hdf_file,'/mov','fr', 200)
    h5writeatt(stitch_hdf_file,'/mov','start_time', 0)
​
end
​
%%
figure(1)
colormap("gray"); 
for j=1:6800
imagesc(stitched_video(:,:,j), [20 200]); drawnow;
end
​
%%
% hdf_roi = stitch_hdf_file;
% h5disp(hdf_roi)