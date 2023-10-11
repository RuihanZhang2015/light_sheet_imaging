​
%this code aligns the layers from two camearas. 
%layer 0 is defined as the first layer captured right after laser ON
​
stitch_dir = 'D:\zebrafish\230820\Fish2_1\stitched';
​
cam1_path = 'D:\zebrafish\230820\Fish2_1\camera1\aligned_layers';
first_cam_time = zeros(30,300); %only read first 300 frames of the mat file to speed up
cam1_alignment = zeros(30,2);
​
%% for camera 1
%load section layer 0 template
for i=0:29
cam_frame = matfile([cam1_path '\layer_' int2str(i) '.mat']);
img_1 = cam_frame.layer_all_uint(:,:,1:300);
first_cam_time(i+1,:) = sum(sum(img_1,1),2)/(256*1280); 
​
end
​
% determine layer 0 : defined at the first layer at laser on
for i=0:29
    cam1_alignment(i+1,1) = i;
    [~, cam1_alignment(i+1,2)] = max(diff(first_cam_time(i+1,:)));
end
​
%% for camera 2
cam2_path = 'D:\zebrafish\230820\Fish2_1\camera2\aligned_layers';
second_cam_time = zeros(30,300);
cam2_alignment = zeros(30,2);
​
%load section layer 0 template
for i=0:29
cam_frame = matfile([cam2_path '\layer_' int2str(i) '.mat']);
img_2 = cam_frame.layer_all_uint(:,:,1:300);
second_cam_time(i+1,:) = sum(sum(img_2,1),2)/(256*1280); 
​
end
​
% determine layer 0 : defined at the first layer at laser on
for i=0:29
    cam2_alignment(i+1,1) = i;
    [~, cam2_alignment(i+1,2)] = max(diff(second_cam_time(i+1,:)));
end
​
%% 
[~, cam1_layer0] = min(diff(cam1_alignment(:,2)))
cam1_layer0 = cam1_layer0 + 1;
[~, cam2_layer0] = min(diff(cam2_alignment(:,2)))
cam2_layer0 = cam2_layer0 + 1;
​
%% this determine the timestamps (in frames) at layer ON event
cam1_rise_time_loc = cam1_alignment(cam1_layer0,2);
cam2_rise_time_loc = cam2_alignment(cam2_layer0,2);
​
%% shift layer index such that they are aligned  
cam1_file_post = circshift([0:29],-1*(cam1_layer0-1))
cam2_file_post = circshift([0:29],-1*(cam2_layer0-1))
​
% save these for use later during stitching 
save([stitch_dir '\cam1_rise_time_loc.mat'], 'cam1_rise_time_loc');
save([stitch_dir '\cam2_rise_time_loc.mat'], 'cam2_rise_time_loc');
save([stitch_dir '\cam1_file_post.mat'], 'cam1_file_post');
save([stitch_dir '\cam2_file_post.mat'], 'cam2_file_post');
​
​
% %% load section layer 0 template
% 
%     first_trial = matfile(['D:\zebrafish\230820\Fish2_1\camera1\aligned_layers\layer_' int2str(18) '.mat']);
%     img_1 = first_trial.layer_all_uint(:,:,400);
% 
%     figure(1)
%     subplot(2,1,1);
%     imagesc(flipud(fliplr(img_1)),[20 200]); colormap("gray"); drawnow;
% 
%     second_trial = matfile(['D:\zebrafish\230820\Fish2_1\camera2\aligned_layers\layer_' int2str(16) '.mat']);
%     img_2 = second_trial.layer_all_uint(:,:,400);
% 
%     figure(1)
%     subplot(2,1,2);
%     imagesc(img_2,[20 200]); colormap("gray"); drawnow;
​
​