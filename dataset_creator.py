import os
import glob

import numpy as np
import sys
import cv2
import h5py
import re

""" clear
This file will create an hp5y file of your dataset.       
This a compressed version of your data that will be able to read in faster. 
In order for this to work you must have your files organized in the following format. 
Falls and Non Falls
- numbered folders containing the video broken down into images
- (ex. Fall1, Fall2, Fall3 ... Fall35... etc) 
folder_location is the name of the folder that contains this dataset 
dset is going to be the compressed hp5y file name of your dataset
"""

# crawls location of drive for all the falls/nonfalls and the images 


folder_location = 'F:\Edits\Thermal' # location of were your dataset is stored 

img_width = 64
img_height = 64
raw = False
dset = 'Thermal_Edit'


def get_dir_lists(dset, folder_location):
    
    path_Fall = folder_location + '\\Fall\\'
    path_ADL = folder_location + '\\NonFall\\'
    vid_dir_list_Fall = glob.glob(path_Fall+'Fall*')
    vid_dir_list_ADL = glob.glob(path_ADL+'NonFall*')
    
    return vid_dir_list_ADL, vid_dir_list_Fall


def init_videos(img_width, img_height, raw, dset, folder_location): 

    '''
    Creates or overwrites h5py group corresponding to root_path (in body), for the h5py file located at 
    'N:/FallDetection/Fall-Data/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height). 
    The h5py group of nested groups is structured as follows:
    
    Params:
        bool raw: if true, data will be not processed (mean centering and intensity scaling)
        int img_wdith: width of images
        int img_height: height of images
        str dset: dataset to be loaded
    '''

    path = '\\Users\Stefan\Documents\lapsum\H5Data\Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height) 

    vid_dir_list_0, vid_dir_list_1 = get_dir_lists(dset, folder_location)
    print(len(vid_dir_list_0))
    print(len(vid_dir_list_1))


    if len(vid_dir_list_0) == 0 and len(vid_dir_list_1) == 0:
        print('no videos found, make sure video files are placed in Fall-Data folde, terminating...')
        sys.exit()

    if raw == False: 
        root_path = dset + '/Processed/Split_by_video'
    else:
        root_path = dset + '/Raw/Split_by_video'

    print('creating data at root_path', root_path)

    def init_videos_helper(root_path): #Nested to keep scope
            with h5py.File(path, 'w') as hf:
                #root_sub = root.create_group('Split_by_video')
                root = hf.create_group(root_path)

                for vid_dir in vid_dir_list_1:
                    init_vid(vid_dir = vid_dir, vid_class = 1, img_width = img_width, img_height = img_height,\
                    hf = root, raw = raw,  dset = dset)

                for vid_dir in vid_dir_list_0: 
                    init_vid(vid_dir = vid_dir, vid_class = 0, img_width = img_width, img_height = img_height, \
                        hf = root, raw = raw,  dset = dset)
    
    if os.path.isfile(path):    
        print("Going down other tree")
        hf = h5py.File(path, 'w')
        if root_path in hf:
            print('video h5py file exists, deleting old group {}, creating new'.format(root_path))
            del hf[root_path]
            hf.close()
            init_videos_helper(root_path)
        else:
            print('File exists, but no group for this data set; initializing..')
            hf.close()
            init_videos_helper(root_path)

    else:#not initialized
        print('No data file exists yet; initializing')
        
        init_videos_helper(root_path)



def init_vid(vid_dir, vid_class, img_width, img_height, hf, raw,  dset):
    '''
    helper function for init_videos. Initialzies a single video.
    Params:
        str vid_dir: path to vid dir of frames to be initialzied
        int vid_class: 1 for Fall, 0 for NonFall
        h5py group: group within which new group is nested
    '''
    vid_dir_name = os.path.basename(vid_dir)
    m = re.search(r'\d+$', vid_dir_name)
    fall_number = int(m.group())
    import pandas as pd
    my_data = pd.read_csv(folder_location + '/Labels.csv')
    current_vid = my_data[my_data.Video == fall_number]
    if len(current_vid) == 0:
        print('Skipping {} as it does not contain a fall'.format(vid_dir_name))
        return
    
    print('-----------------------')
    print('initializing vid at', vid_dir_name)
    raw = True
    sort = True
    fpath = vid_dir
    data = create_img_data_set(fpath, img_width, img_height, raw, sort, dset)
    print("Creating at", vid_dir_name)
    grp = hf.create_group(vid_dir_name)
    labels = np.zeros(len(data))
    if vid_class == 1:
        labels = get_fall_indeces(vid_dir_name,labels, dset)
        print(np.unique(labels, return_counts=True))
    else:
        print("Non fall thus labels are 0ed")
    
    grp['Labels'] = labels
    grp['Data'] = data

def get_fall_indeces(Fall_name, labels, dset):
    
    m = re.search(r'\d+$', Fall_name)
    fall_number = int(m.group())

    import pandas as pd
    my_data = pd.read_csv(folder_location + '/Labels.csv')
    
    current_vid = my_data[my_data.Video == fall_number]
    print(current_vid)
    if len(current_vid) == 0:
        return

    if len(current_vid) == 1:
        print("Single Fall")
        labels[int(current_vid.Start.iloc[0]):int(current_vid.Stop.iloc[0])] = 1
    else:
        print("Two Falls")
        labels[int(current_vid.Start.iloc[0]):int(current_vid.Stop.iloc[0])] = 1
        labels[int(current_vid.Start.iloc[1]):int(current_vid.Stop.iloc[1])] = 1

    return labels
        


def create_img_data_set(fpath, ht = 128, wd = 128, raw = False, sort = True, dset = 'Thermal'):
        '''
        Creates data set of all images located at fpath. Sorts images
        Params:
            str fpath: path to images to be processed
            bool raw: if True does mean centering and rescaling 
            bool sort: if True, sorts frames, ie. keeps sequential order, which may be lost due to glob
            dset: dataset
        Returns:
            ndarray data: Numpy array of images at fpath. Shape (samples, img_width*img_height),
            samples isnumber of images at fpath.
        '''
        
        #print('gathering data at', fpath)
        fpath = fpath.replace('\\', '/')
        # print(fpath+'/*.png')
        frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')
        frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        #print("\n".join(frames)) #Use this to check if sorted

        data=np.zeros((frames.__len__(),ht,wd,1))
        for x,i in zip(frames, range(0,frames.__len__())):
            #print(x,i)
            img=cv2.imread(x,0) #Use this for RGB to GS
            #print('x', x)
            #img=cv2.imread(x,-1) #Use this for loading as is(ie. 16 bit needs this, else gets converted to 8)
            # print('img.shape', img.shape)
            img=cv2.resize(img,(ht,wd))#resize
            img=img.reshape(ht,wd,1)

            img=img-np.mean(img)#Mean centering
            img=img.astype('float32') / 255 #rescaling

            data[i,:,:,:]=img

        print('data.shape', data.shape)
        return data

def flip_windowed_arr(windowed_data):
    """
    windowed_data: of shape (samples, win_len,...)
    
    returns shape len(windowed_data), win_len, flattened_dim)
    Note: Requires openCV
    """
    win_len = windowed_data.shape[1]
    flattened_dim = np.prod(windowed_data.shape[2:])
    #print(flattened_dim)
    flipped_data_windowed = np.zeros((len(windowed_data), win_len, flattened_dim)) #Array of windows
    print(flipped_data_windowed.shape)
    i=0
    for win_idx in range(len(windowed_data)):
        window = windowed_data[win_idx]
        flip_win = np.zeros((win_len, flattened_dim))

        for im_idx in range(len(window)):
            im = window[im_idx]
            hor_flip_im = cv2.flip(im,1)
            #print(hor_flip_im.shape)
            #print(flip_win[im_idx].shape)
            
            flip_win[im_idx] = hor_flip_im.reshape(flattened_dim)
            
        flipped_data_windowed[win_idx] = flip_win
    return flipped_data_windowed

init_videos(img_width, img_height, raw, dset, folder_location)