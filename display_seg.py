import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import tifffile
import skimage.exposure as exposure

DAPI_CHANNEL_INDEX = 0
SEG_DIR = 'seg'

# TODO(NA): import this from utils
def load_channel(tif_paths, channel_idx):
    ''' load the desired channel as a 2d np array
    
    Args:
        - tif_paths: list of strs, the path to target multi-channel tif files
        - channel_idx: int, the 0-indexing position of the desired 
                       channel (DAPI being usually 0th)
        - if_show_images: bool, visualize the loaded image channel if set to true
        
    Returns:
        - imgs: list of 2d np array, the desired channel of the image
    '''
    imgs = []
    for path in tif_paths:
        
        # upon being loaded, a tif file is usually a 3d np array of size
        # (n_channels x width x height)
        channel = tifffile.imread(path)[channel_idx, :, :]
        channel = channel.reshape(channel.shape[0], channel.shape[1])
        
        # re-scale the image intensity to a range of 0-255
        channel = exposure.rescale_intensity(channel, in_range='image', out_range=(0,255)).astype(np.uint8)
        imgs.append(channel)
    
    return imgs

def main():
    cd = os.path.dirname(os.path.abspath(__file__))
    tif_paths = list(pathlib.Path('{}/tif'.format(cd)).glob('*.tif'))
    dapi_channels = load_channel(tif_paths, DAPI_CHANNEL_INDEX)

    for i, chan in enumerate(dapi_channels):
        print(i, chan)
        plt.subplot(1, 2, 1)
        plt.imshow(chan)

        seg_name = tif_paths[i].name.rstrip('.tif') + '_seg.npy'
        cd = os.path.dirname(os.path.abspath(__file__))
        print('{}/seg/{}'.format(cd, seg_name))
        seg = np.load('{}/seg/{}'.format(cd, seg_name))
        plt.subplot(1,2,2)
        plt.imshow(seg)
        
        plt.show()

if __name__ == "__main__":
    main()
