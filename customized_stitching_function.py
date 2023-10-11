# def customized_function(vol1,vol2):
#     return np.concatenate((vol1,vol2),axis=1)

import numpy as np
def stitch2cam_0826fish5(im1, im2):
    interp_start_pos = 12
    shift = 7
    
    # Rotate im2 image
    im2 = np.flip(np.flip(im2, 0), 1)
    
    # Convert to float (equivalent of single in MATLAB)
    im1 = im1[4:].astype(float)
    im2 = im2[:-4].astype(float)
    imwidth = im1.shape[1]
    x_old = np.arange(imwidth)
    x_new = np.linspace(interp_start_pos, imwidth, imwidth)
    
    #  Nearest-neighbor interpolation
    indices = np.clip(np.round((x_new - x_old[0]) / (x_old[1] - x_old[0])).astype(int), 0, imwidth - 1)
    tmp2 = im2[:, indices].T
    
    # Shift (equivalent of circshift in MATLAB)
    tmp2 = np.roll(tmp2, shift, axis=1)
    tmp2[:, :shift] = 31
    
    # Concatenate (equivalent of cat in MATLAB)
    im = np.vstack((tmp2, im1)).astype(np.uint16)
    return im
    
# Assuming you have the images im1 and im2 loaded, you can call the function as:
# result = stitch2cam_0826fish5(im1, im2)