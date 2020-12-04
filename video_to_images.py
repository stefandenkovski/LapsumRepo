import cv2
import os
import glob

# Opens the Video file
path = '/DataFolders/'
list_of_files = []
list_of_folders = []
file_count = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.mp4'):
            if filename.startswith('._'):
                break
            list_of_files.append(os.sep.join([dirpath, filename]))
            list_of_folders.append(dirpath)
            file_count += 1
print(list_of_files)
print(list_of_folders)
print(file_count)

for video in range(file_count):
    cap = cv2.VideoCapture(list_of_files[video])
    frame_locations = list_of_folders[video] + "/frames"
    if not os.path.exists(frame_locations):
        os.makedirs(frame_locations)
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(frame_locations, 'frame' + str(i) + '.jpg'), frame)
            i += 1

    cap.release()
    cv2.destroyAllWindows()
    print('{0:.3f}%'.format((video / file_count * 100)))