%metadata alignment
clear all;
​
for jjj = 0:5
offset_index = jjj*5
%offset_index = 0; 
​
% Sample Data
% Parse XML
xDoc = xmlread('E:\20230826_gal4_3xPosi2_5-6dpf_xcaspr_F2_40us_4980us_UV\raw\camera1\nrrds\fish5_2.xiseq');
allListItems = xDoc.getElementsByTagName('file');
frameNumbers = zeros(1, allListItems.getLength);
% Extract frame numbers
for k = 0:allListItems.getLength-1
    thisListItem = allListItems.item(k);
    frameNumbers(k+1) = str2double(thisListItem.getAttribute('frame'));
end
​
%% read nrrd file
​
%add path to common functions
addpath('D:\zebrafish\common_func')
save_path = 'E:\20230826_gal4_3xPosi2_5-6dpf_xcaspr_F2_40us_4980us_UV\Fish5_2\camera1\aligned_layers';
raw_path = 'E:\20230826_gal4_3xPosi2_5-6dpf_xcaspr_F2_40us_4980us_UV\raw\camera1\nrrds\fish5_2_';
total_desired_frames = 6900;
​
​
%%
clear all_vid;
video_initial_drops = 30; %discard the first 30 volumns
all_vid = zeros(256,1280,70000,'uint16');
nrrd_vec = [1:7];
for i=1:7
    [raw_path int2str(nrrd_vec(i)) '.nrrd']
    all_vid(:,:,(i-1)*10000+1:i*10000) = permute(nrrdread([raw_path int2str(nrrd_vec(i)) '.nrrd']),[2 1 3]);
​
end
​
​
%%
all_vid = all_vid(:,:,video_initial_drops*30+1:end);
frameNum_Short = frameNumbers(video_initial_drops*30+1:end);
​
%%
section_1_len = size(all_vid,3); %total length of all_vid
frameNum_sec1 = frameNum_Short(1:section_1_len);
frameNumberMod_sec1 = rem(frameNum_sec1,30);
​
% take the frame numbers, find all the ones with same reminder
[~, frame_index] = find(frameNumberMod_sec1== (offset_index+0)); %0
[~, miss_frame_loc] = find(diff(frame_index)>30)
layer_1_sec1 = all_vid(:,:,frame_index);
layer_1_sec1 = insert_miss_frame(layer_1_sec1,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec1== (offset_index+1)); %1
[~, miss_frame_loc] = find(diff(frame_index)>30)
layer_2_sec1 = all_vid(:,:,frame_index);
layer_2_sec1 = insert_miss_frame(layer_2_sec1,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec1== (offset_index+2)); %2
[~, miss_frame_loc] = find(diff(frame_index)>30)
layer_3_sec1 = all_vid(:,:,frame_index);
layer_3_sec1 = insert_miss_frame(layer_3_sec1,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec1== (offset_index+3)); %3
[~, miss_frame_loc] = find(diff(frame_index)>30)
layer_4_sec1 = all_vid(:,:,frame_index);
layer_4_sec1 = insert_miss_frame(layer_4_sec1,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec1== (offset_index+4)); %4
[~, miss_frame_loc] = find(diff(frame_index)>30)
layer_5_sec1 = all_vid(:,:,frame_index);
layer_5_sec1 = insert_miss_frame(layer_5_sec1,miss_frame_loc);
​
%%
​
% sum_line = zeros(1,size(layer_1_sec2,3));
%
% for i=1:size(layer_1_sec2,3)
%     sum_line(i) = sum(sum(layer_1_sec2(100:200,600:1000,i)))/(length(100:200)*length(600:1000));
% end
%
% figure(1); plot(sum_line(1:end)); drawnow;
​
​
%% section 2
clear all_vid;
all_vid = zeros(256,1280,70000,'uint16');
nrrd_vec = [8:14];
for i=1:7
    disp([raw_path int2str(nrrd_vec(i)) '.nrrd'])
    all_vid(:,:,(i-1)*10000+1:i*10000) = permute(nrrdread([raw_path int2str(nrrd_vec(i)) '.nrrd']),[2 1 3]);
​
end
​
​
%%
section_2_len = size(all_vid,3); %total length of all_vid
​
frameNum_sec2 = frameNum_Short(section_1_len+1 : section_1_len + section_2_len);
frameNumberMod_sec2 = rem(frameNum_sec2,30);
​
% take the frame numbers, find all the ones with same reminder
[~, frame_index] = find(frameNumberMod_sec2== (offset_index + 0));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_1_sec2 = all_vid(:,:,frame_index);
layer_1_sec2 = insert_miss_frame(layer_1_sec2,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec2== (offset_index + 1));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_2_sec2 = all_vid(:,:,frame_index);
layer_2_sec2 = insert_miss_frame(layer_2_sec2,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec2== (offset_index + 2));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_3_sec2 = all_vid(:,:,frame_index);
layer_3_sec2 = insert_miss_frame(layer_3_sec2,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec2== (offset_index + 3));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_4_sec2 = all_vid(:,:,frame_index);
layer_4_sec2 = insert_miss_frame(layer_4_sec2,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec2== (offset_index + 4));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_5_sec2 = all_vid(:,:,frame_index);
layer_5_sec2 = insert_miss_frame(layer_5_sec2,miss_frame_loc);
​
%% section 3
clear all_vid;
all_vid = zeros(256,1280,70000,'uint16');
nrrd_vec = [15:21];
for i=1:7
    disp([raw_path int2str(nrrd_vec(i)) '.nrrd']);
    all_vid(:,:,(i-1)*10000+1:i*10000) = permute(nrrdread([raw_path int2str(nrrd_vec(i)) '.nrrd']),[2 1 3]);
​
end
​
%%
section_3_len = size(all_vid,3); %total length of all_vid
​
frameNum_sec3 = frameNum_Short(section_1_len + section_2_len + 1 : end);
frameNumberMod_sec3 = rem(frameNum_sec3,30);
​
​
% take the frame numbers, find all the ones with same reminder
[~, frame_index] = find(frameNumberMod_sec3== (offset_index + 0));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_1_sec3 = all_vid(:,:,frame_index);
layer_1_sec3 = insert_miss_frame(layer_1_sec3,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec3== (offset_index + 1));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_2_sec3 = all_vid(:,:,frame_index);
layer_2_sec3 = insert_miss_frame(layer_2_sec3,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec3== (offset_index + 2));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_3_sec3 = all_vid(:,:,frame_index);
layer_3_sec3 = insert_miss_frame(layer_3_sec3,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec3== (offset_index + 3));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_4_sec3 = all_vid(:,:,frame_index);
layer_4_sec3 = insert_miss_frame(layer_4_sec3,miss_frame_loc);
​
[~, frame_index] = find(frameNumberMod_sec3== (offset_index + 4));
[~, miss_frame_loc] = find(diff(frame_index)>30) %get all the missing frame locations
layer_5_sec3 = all_vid(:,:,frame_index);
layer_5_sec3 = insert_miss_frame(layer_5_sec3,miss_frame_loc);
​
%% verify
figure(3)
subplot(3,1,1)
imagesc(layer_1_sec1(:,:,400),[20 200]); colormap(gray); drawnow;
subplot(3,1,2)
imagesc(layer_1_sec2(:,:,1),[20 200]); colormap(gray); drawnow;
subplot(3,1,3)
imagesc(layer_1_sec3(:,:,1),[20 200]); colormap(gray); drawnow;
​
​
%% combine all 3 traces
clear all_vid;
maxframe_sec1 = size(layer_1_sec1,3);
maxframe_sec2 = size(layer_1_sec2,3);
maxframe_sec3 = size(layer_1_sec3,3);
​
layer_1_all = zeros(256,1280,maxframe_sec1+maxframe_sec2+maxframe_sec3);
layer_1_all(:,:,1:maxframe_sec1) = layer_1_sec1;
layer_1_all(:,:,maxframe_sec1+1 : maxframe_sec1+maxframe_sec2) = layer_1_sec2;
layer_1_all(:,:,maxframe_sec1+maxframe_sec2+1 : end) = layer_1_sec3;
​
maxframe_sec1 = size(layer_2_sec1,3);
maxframe_sec2 = size(layer_2_sec2,3);
maxframe_sec3 = size(layer_2_sec3,3);
​
layer_2_all = zeros(256,1280,maxframe_sec1+maxframe_sec2+maxframe_sec3);
layer_2_all(:,:,1:maxframe_sec1) = layer_2_sec1;
layer_2_all(:,:,maxframe_sec1+1 : maxframe_sec1+maxframe_sec2) = layer_2_sec2;
layer_2_all(:,:,maxframe_sec1+maxframe_sec2+1 : end) = layer_2_sec3;
​
maxframe_sec1 = size(layer_3_sec1,3);
maxframe_sec2 = size(layer_3_sec2,3);
maxframe_sec3 = size(layer_3_sec3,3);
​
layer_3_all = zeros(256,1280,maxframe_sec1+maxframe_sec2+maxframe_sec3);
layer_3_all(:,:,1:maxframe_sec1) = layer_3_sec1;
layer_3_all(:,:,maxframe_sec1+1 : maxframe_sec1+maxframe_sec2) = layer_3_sec2;
layer_3_all(:,:,maxframe_sec1+maxframe_sec2+1 : end) = layer_3_sec3;
​
maxframe_sec1 = size(layer_4_sec1,3);
maxframe_sec2 = size(layer_4_sec2,3);
maxframe_sec3 = size(layer_4_sec3,3);
​
layer_4_all = zeros(256,1280,maxframe_sec1+maxframe_sec2+maxframe_sec3);
layer_4_all(:,:,1:maxframe_sec1) = layer_4_sec1;
layer_4_all(:,:,maxframe_sec1+1 : maxframe_sec1+maxframe_sec2) = layer_4_sec2;
layer_4_all(:,:,maxframe_sec1+maxframe_sec2+1 : end) = layer_4_sec3;
​
maxframe_sec1 = size(layer_5_sec1,3);
maxframe_sec2 = size(layer_5_sec2,3);
maxframe_sec3 = size(layer_5_sec3,3);
​
layer_5_all = zeros(256,1280,maxframe_sec1+maxframe_sec2+maxframe_sec3);
layer_5_all(:,:,1:maxframe_sec1) = layer_5_sec1;
layer_5_all(:,:,maxframe_sec1+1 : maxframe_sec1+maxframe_sec2) = layer_5_sec2;
layer_5_all(:,:,maxframe_sec1+maxframe_sec2+1 : end) = layer_5_sec3;
​
%%
sum_line = zeros(1,size(layer_1_all,3));
​
for i=1:size(layer_1_all,3)
    sum_line(i) = sum(sum(layer_1_all(100:200,600:1000,i)))/(length(100:200)*length(600:1000));
end
​
figure(1); plot(sum_line(1:end)); drawnow;
​
​
​
%%
layer_all_uint = uint16(layer_1_all(:,:,1:total_desired_frames));
save([save_path '\layer_' int2str(offset_index+0) '.mat'], 'layer_all_uint' ,'-v7.3');
​
layer_all_uint = uint16(layer_2_all(:,:,1:total_desired_frames));
save([save_path '\layer_' int2str(offset_index+1) '.mat'], 'layer_all_uint' ,'-v7.3');
​
layer_all_uint = uint16(layer_3_all(:,:,1:total_desired_frames));
save([save_path '\layer_' int2str(offset_index+2) '.mat'], 'layer_all_uint' ,'-v7.3');
​
layer_all_uint = uint16(layer_4_all(:,:,1:total_desired_frames));
save([save_path '\layer_' int2str(offset_index+3) '.mat'], 'layer_all_uint' ,'-v7.3');
​
layer_all_uint = uint16(layer_5_all(:,:,1:total_desired_frames));
save([save_path '\layer_' int2str(offset_index+4) '.mat'], 'layer_all_uint' ,'-v7.3');
​
clearvars -except jjj
​
​
end