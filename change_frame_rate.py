from inspect import currentframe
import cv2
import os
import glob
import shutil
import math
import re

dset = 'IP'
path = 'F:/{}/'.format(dset)

# Ip _ 1280, ONI 1218, ZED 1303, Thermal - 1250

# get the list of falls and nonfall folders 
current_frame_rate = 20
desired_frame_rate = 8 
adjustment_factor = current_frame_rate/desired_frame_rate
print(adjustment_factor)



path_Fall = path + '\\Fall\\'
path_ADL = path + '\\NonFall\\'
list_Falls = glob.glob(path_Fall+'Fall*')
list_ADLs = glob.glob(path_ADL+'NonFall*')

def interpolate_frames(fpath):
    
    # Get list of exisits images
    print("Moving: ", fpath )
    old_frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')
    old_frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    # Define new locations, create directors, and then move files
    new_location = fpath.replace('F:/', 'F:/Edits/')

    count = 0
    frame = 0 
    for i, old_frame in enumerate(old_frames):
        if i == math.floor(count):
            if not os.path.exists(new_location):
                os.makedirs(new_location)
            if os.path.exists(new_location):
                shutil.copyfile(old_frame, new_location + '/frame' + str(frame) + old_frame[-4:])
                frame += 1
            count += adjustment_factor
    
for Fall in list_Falls:
    interpolate_frames(Fall)

for NonFall in list_ADLs:
    interpolate_frames(NonFall)
    
