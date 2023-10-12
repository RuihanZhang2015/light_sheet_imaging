from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def stitch2cam_0826fish5(
        im1, 
        im2,
        width = 1280
        ):

    # Flip the axis of the image
    im2 = np.flip(np.flip(im2, 0), 1)
    
    # Crop
    im1 = im1[4:,:] 
    im2 = im2[:-4,12:]

    # Resize
    im2_pil = Image.fromarray(im2)
    im2_pil = im2_pil.resize((width, 252), Image.Resampling.NEAREST)
    im2 = np.asarray(im2_pil)

    # Shift
    shift = 7
    img2_final = np.zeros((252,width))
    img2_final[:,shift:] = im2[:,:-shift]

    # Stitch
    stitched = np.concatenate((im2,im1),axis=0)
    return stitched.astype(np.uint16)
    