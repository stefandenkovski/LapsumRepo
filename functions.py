import h5py
import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import roc_auc_score
import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import re
from io import StringIO
import ffmpeg
import pdb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    f1_score,
    auc,
    precision_recall_curve,
)
import pdb

"""
 Function that takes inputs (sample, reconstruction, and label)
 generates reconstruction erros, and then generates performance metrics and saves these in a csv

performance metrics
- AUC ROC and PR for std, and mean of frame error (both day and night)
- AUC ROC and PR for std and mean of window error for different thresholds (both day and night)

"""

def get_performance_metrics(sample, output, labels, window_len):
    window_std, window_mean, window_labels = get_window_metrics(sample, output, labels, window_len)
    frame_std, frame_mean, frame_labels = get_frame_metrics(sample, output, labels, window_len)
    return(frame_std, frame_mean, frame_labels, window_std, window_mean, window_labels)



def get_window_metrics(sample, output, labels, window_len):
    recon_data = output.reshape(output.shape[1], window_len, 64 * 64)
    sample_data = sample.reshape(sample.shape[1], window_len, 64 * 64)
    labels = shape_labels(labels)

    recon_error = np.mean(np.power(sample_data - recon_data, 2), axis=2)
    mean_window_error = []
    std_window_error = [] 
    window_labels = [] 
    for tolerance in range(1, window_len+1):
        stride = 1
        windowed_labels = create_windowed_labels(labels, stride, tolerance, window_len)
        windowed_labels = windowed_labels[:,0]
        inwin_mean = np.mean(recon_error, axis =1)
        inwin_std = np.std(recon_error, axis =1)
        mean_window_error.append(inwin_mean)
        std_window_error.append(inwin_std)
        window_labels.append(windowed_labels)
    return(mean_window_error, std_window_error, window_labels)

def get_frame_metrics(sample, output, labels, window_len):
    recon_data = output.reshape(output.shape[1], window_len, 64 * 64)
    sample_data = sample.reshape(sample.shape[1], window_len, 64 * 64)
    labels = shape_labels(labels)

    recon_error = np.mean(np.power(sample_data - recon_data, 2), axis=2)

    # ------- Frame Reconstruction Error ---------------
    # create empty matrix w/ orignal number of frames
    mat = np.zeros((len(recon_error) + window_len - 1, len(recon_error)))
    mat[:] = np.NAN
    # dynmaically fill matrix with windows values for each frame
    #print(len(recon_error))
    for i in range(len(recon_error)):
        win = recon_error[i]
        #print(len(win))
        #print(i)
        mat[i : len(win) + i, i] = win
    frame_scores = []
    # each row corresponds to a frame across windows
    # so calculate stats for a single frame frame(row)
    for i in range(len(mat)):
        row = mat[i, :]
        mean = np.nanmean(row, axis=0)
        std = np.nanstd(row, axis=0)
        frame_scores.append((mean, std, mean + std * 10 ** 3))
    
    frame_scores = np.array(frame_scores)
    x_std = frame_scores[:, 1]
    x_mean = frame_scores[:, 0]

    return(x_mean, x_std, labels)



def get_total_performance_metrics(frame_stats, window_stats, window_len):

    video_metrics = np.zeros((len(frame_stats), 5, window_len ))
    print(video_metrics.shape)

    # here i need to get the error and everything and store it in video metrics 

    for i in range(len(frame_stats)):
        # print(i)
        # this a single video metrics
        frame_mean, frame_std, frame_labels = frame_stats[i]
        mean_window_error, std_window_error, window_labels = window_stats[i]
    
        # store frame results for this vidoe
        video_metrics[i, 0, 0], video_metrics[i, 1, 0], video_metrics[i, 2, 0], video_metrics[i, 3, 0] = get_performance_values(frame_mean, frame_std, frame_labels)
        video_metrics[i, 4, 0] = 0 
        # store each thresholds results for this video 
        for j in range(1, window_len):
            vid_labels = window_labels[j-1]
            #print(vid_labels)
            if len(np.unique(vid_labels)) != 2:
                # print("ERROR: no fall for threshold {} in video {}".format(j, i))
                continue
            video_metrics[i, 0, j], video_metrics[i, 1, j], video_metrics[i, 2, j], video_metrics[i, 3, j] = get_performance_values(mean_window_error[j-1], std_window_error[j-1], vid_labels)
            video_metrics[i, 4, j] = j 
    
    print('saving')

    video_metrics[video_metrics == 0] = np.nan
    final_performance = np.nanmean(video_metrics, axis=0) # get the mean performance across all videos 

    #np.savetxt('results.csv', final_performance, delimiter=',', fmt='%d')
    pd.DataFrame(final_performance).to_csv("results.csv")


def get_performance_values(vid_mean, vid_std, vid_labels):

    # calculate metrics for Standard Deviation 
    fpr, tpr, thresholds = roc_curve(y_true=vid_labels[:len(vid_std)], y_score=vid_std, pos_label=1)
    std_AUROC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(vid_labels[:len(vid_std)], vid_std)
    std_AUPR = auc(recall, precision)

    # calculate the Mean AUC
    fpr, tpr, thresholds = roc_curve(y_true=vid_labels[:len(vid_std)], y_score=vid_mean, pos_label=1)
    mean_AUROC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(vid_labels[:len(vid_std)], vid_mean)
    mean_AUPR = auc(recall, precision)
    #print(std_AUROC, std_AUPR, mean_AUROC, mean_AUPR)
    return std_AUROC, mean_AUROC, std_AUPR, mean_AUPR






def shape_labels(labels):
    # generate labels
    label = labels[0, :, :]
    windowed_labels = label # shape (window_len, # of windows) 
    frame_labels = un_window(label)
    return frame_labels



def create_windowed_labels(labels, stride, tolerance, window_length):
    output_length = int(np.floor((len(labels) - window_length) / stride))+1
    output_shape = (output_length, 1)
    total = np.zeros(output_shape)
    i=0
    while i < output_length:
        next_chunk = np.array([labels[i+j] for j in range(window_length)])
        num_falls = sum(next_chunk) #number of falls in the window

        if num_falls >= tolerance:
            total[i] = 1
        else:
            total[i] = 0

        i = i+stride
    labels_windowed = total
    return labels_windowed

def un_window(windowed_data):
# Input: Windowed Data with format (window_length, # of windows )
    unwindowed_data = np.zeros(windowed_data.shape[0] + windowed_data.shape[1])
    for i in range(len(unwindowed_data)):
        if i >= windowed_data.shape[1]:
            last_window = windowed_data[:, i - 1]
            unwindowed_data[i:] = last_window
            break
        else:
            unwindowed_data[i] = windowed_data[0, i]

    return unwindowed_data

def animate(test_data, recons_seq, frame_mean, dset, start_time):
    ani_dir = "./Animation/{}/".format(dset)
    ani_dir = ani_dir + "/{}".format(start_time)
    if not os.path.isdir(ani_dir):
        os.makedirs(ani_dir)
    print("saving animation to {}".format(ani_dir))

    animate_fall_detect_present(
        testfall=test_data[:, 0, :].reshape(len(test_data), 64, 64, 1),
        recons=recons_seq[:, 0, :].reshape(len(recons_seq), 64, 64, 1),
        win_len=1,
        scores=frame_mean,
        to_save=ani_dir + "/{}.mp4".format(len(test_data)),
    )


def animate_fall_detect_present(
    testfall, recons, scores, win_len, threshold=0, to_save="./test.mp4"
):
    """
    Pass in data for single video, recons is recons frames, scores is x_std or x_mean etc.
    Threshold is RRE, mean, etc..
    """
    import matplotlib.gridspec as gridspec

    Writer = animation.writers["pillow"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])


    ht, wd = 64, 64

    eps = 0.0001
    # setup figure
    # fig = plt.figure()
    fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(6, 6))

    ax1.axis("off")
    ax3.axis("off")
    # ax1=fig.add_subplot(2,2,1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax2=fig.add_subplot(gs[-1,0])
    ax2 = fig.add_subplot(gs[1, :])

    # ax2.set_yticks([])
    # ax2.set_xticks([])
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Frame")

    if threshold != 0:
        ax2.axhline(y=threshold, color="r", linestyle="dashed", label="RRE")
        ax2.legend()

    # ax3=fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title("Reconstruction")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # set up list of images for animation
    ims = []

    for time in range(1, len(testfall) - (win_len - 1) - 1):
        im1 = ax1.imshow(testfall[time].reshape(ht, wd), cmap="gray", aspect="equal")
        figure = recons[time].reshape(ht, wd)
        im2 = ax3.imshow(figure, cmap="gray", aspect="equal")
        # print(im1.shape)
        # print(im2.shape)

        # print("time={} mse={} std={}".format(time,mse_difficult[time],std))
        if time > 0:

            scores_curr = scores[0:time]

            fall_pts_idx = np.argwhere(scores_curr > threshold)
            nonfall_pts_idx = np.argwhere(scores_curr <= threshold)

            fall_pts = scores_curr[fall_pts_idx]
            nonfall_pts = scores_curr[nonfall_pts_idx]

            if fall_pts_idx.shape[0] > 0:
                # pass
                (plot_r,) = ax2.plot(fall_pts_idx, fall_pts, "r.")
                (plot,) = ax2.plot(nonfall_pts_idx, nonfall_pts, "b.")
            else:

                (plot,) = ax2.plot(scores_curr, "b.")

        else:
            (plot,) = ax2.plot(scores[0], "b.")
            (plot_r,) = ax2.plot(scores[0], "b.")

        ims.append([im1, plot, im2, plot_r])  # list of ims

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=40, repeat=False)
    # plt.tight_layout()
    #plt.show()
    # gs.tight_layout(fig)
    ani.save(to_save)

    ani.event_source.stop()
    del ani
    plt.close()
    # plt.show()
    # return ani

def create_pytorch_dataset(name, dset, path, window_len, fair_compairson, stride):
    falls = []
    adl = []
    if fair_compairson == True:
        shared_adl_vids = np.loadtxt('shared_adl_vids.txt').astype(int)
        shared_fall_vids = np.loadtxt('shared_fall_vids.txt').astype(int)
        print(shared_fall_vids)
        # create list of all fall and nonfall folders
        for (root, dirs, files) in os.walk("F:/{}/Fall".format(dset)):
            for dir in dirs:
                x = re.findall('[0-9]+', dir)[0]
                if int(x) in shared_fall_vids:
                    falls.append(dir)
        
        for (root, dirs, files) in os.walk("F:/{}/NonFall".format(dset)):
            for dir in dirs:
                x = re.findall('[0-9]+', dir)[0]
                if int(x) in shared_adl_vids:
                    adl.append(dir)
        print(falls)
        print(adl)
    elif fair_compairson == False:
        # create list of all fall and nonfall folders
        for (root, dirs, files) in os.walk("F:/{}/Fall".format(dset)):
            if len(dirs) > 0:
                falls.extend(dirs)
        for (root, dirs, files) in os.walk("F:/{}/NonFall".format(dset)):
            if len(dirs) > 0:
                adl.extend(dirs)
        print(falls)
        print(adl)
        

    x_data_fall = []
    y_data_fall = []
    x_data_adl = []
    y_data_adl = []
    x_info_fall = []
    x_info_adl = []

    # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

    # load in images of falls
    with h5py.File(path, "r") as hf:
        data_dict = hf["{}/Processed/Split_by_video".format(name)]
        # print(data_dict.keys())
        for Fall_name in falls:
            try:
                vid_total = data_dict[Fall_name]["Data"][:]
                if len(vid_total) < 100:
                    continue
                x_data_fall.append(vid_total)
                x_info_fall.append(Fall_name) #[4:]
                labels_total = data_dict[Fall_name]["Labels"][:]
                y_data_fall.append(labels_total)
            except:
                print("Skipped", Fall_name)
        for adl_name in adl:
            try:
                vid_total = data_dict[adl_name]["Data"][:]
                if len(vid_total) < 100:
                    continue
                x_data_adl.append(vid_total)
                x_info_adl.append(adl_name) #[7:]
                labels_total = data_dict[adl_name]["Labels"][:]
                y_data_adl.append(labels_total)
            except:
                print("Skipped", Fall_name)


    print(len(x_data_fall))
    print(len(y_data_fall))
    print(len(x_data_adl))
    print(len(y_data_adl))
    # get alll falls in data set
    print(x_info_adl)
    print(x_info_fall)
    # get matching day/night label from falls
    labels_dir = "F:/{}/".format(dset) + "Labels.csv"
    my_data = pd.read_csv(labels_dir)
    # sorting by first name
    my_data.sort_values("Video", inplace=True)
    my_data.drop_duplicates(subset="Video", keep="first", inplace=True)
    print(my_data.head())


    # pdb.set_trace()
    # %%    temp_df = my_data.loc[my_data["Video"] == int(fall), "ToD"]

    # ----------------------------------------------------------------------------
    # *** PREPARING DATASET LOADER ***
    # ----------------------------------------------------------------------------

    # 1) Need a ADL loader and a Fall Loader

    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, data, window):
            "Initialization"
            self.labels = labels
            self.data = data
            self.window = window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data)

        def __getitem__(self, index):
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_list = []
            Y_list = []
            # load a single video to chop up into windows
            ind_vid = self.data[index]
            ind_label = self.labels[index]
            # loop through each frame of the video (stopping window length short)
            for i in range(len(ind_vid) - self.window):
                # select the current window of the video
                X = ind_vid[i : i + self.window]
                y = ind_label[i : i + self.window]
                # add the current window the list of windows
                X_list.append(X)
                Y_list.append(y)
            # convert lists into arrays with proper size
            X = np.vstack(X_list)
            X = np.reshape(X_list, (len(ind_vid) - self.window, self.window, 64, 64))
            y = np.vstack(Y_list).T
            # X should be (window-length, 64, 64, # of windows w/in video) array
            # ex. (8, 64, 64, 192) for a 200 frame video and window size of 8
            # y is array (8, # of windows w/in video)
            return X, y


    Test_Dataset = Dataset(y_data_fall, x_data_fall, window=window_len)
    test_dataloader = data.DataLoader(Test_Dataset, batch_size=1)

    Train_Dataset = Dataset(y_data_adl, x_data_adl, window=window_len)
    train_dataloader = data.DataLoader(Train_Dataset, batch_size=1)

    return(Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)



def create_multimodal_pytorch_dataset(name, dset, path, window_len, stride):
    
    def load_data_set(dset, name, path):
        falls = []
        adl = []
        print(dset)
        print(name)
        print(path)
        # create list of all fall and nonfall folders
        for (root, dirs, files) in os.walk("F:/{}/Fall".format(dset)):
            if len(dirs) > 0:
                falls.extend(dirs)
        for (root, dirs, files) in os.walk("F:/{}/NonFall".format(dset)):
            if len(dirs) > 0:
                adl.extend(dirs)

        x_data_fall = []
        y_data_fall = []
        x_data_adl = []
        y_data_adl = []
        x_info_fall = []
        x_info_adl = []
        
        # load in images of falls
        with h5py.File(path, "r") as hf:
            data_dict = hf["{}/Processed/Split_by_video".format(name)]
            # print(data_dict.keys())
            for Fall_name in falls:
                try:
                    vid_total = data_dict[Fall_name]["Data"][:]
                    if len(vid_total) < 100:
                        continue
                    x_data_fall.append(vid_total)
                    x_info_fall.append(Fall_name[4:])
                    labels_total = data_dict[Fall_name]["Labels"][:]
                    y_data_fall.append(labels_total)
                except:
                    print("Skipped", Fall_name)
            for adl_name in adl:
                try:
                    vid_total = data_dict[adl_name]["Data"][:]
                    if len(vid_total) < 100:
                        continue
                    x_data_adl.append(vid_total)
                    x_info_adl.append(adl_name[7:])
                    labels_total = data_dict[adl_name]["Labels"][:]
                    y_data_adl.append(labels_total)
                except:
                    print("Skipped", adl_name)

        print(len(x_data_fall))
        print(len(y_data_fall))
        print(len(x_data_adl))
        print(len(y_data_adl))
        '''
        # get matching day/night label from falls
        labels_dir = "F:/{}/".format(dset) + "Labels.csv"
        my_data = pd.read_csv(labels_dir)
        # sorting by first name
        my_data.sort_values("Video", inplace=True)
        my_data.drop_duplicates(subset="Video", keep="first", inplace=True)
        x_info_fall_ToD = []
        for fall in x_info_fall:
            temp_df = my_data.loc[my_data["Video"] == int(fall), "ToD"]
            x_info_fall_ToD.append(temp_df.values.astype(int))
        x_info_fall_ToD = np.vstack(x_info_fall_ToD)
        np.savetxt(
            "results/Edits/{}/night_time_{}.csv".format(str(name) + str(name), start_time),
            x_info_fall_ToD,
            delimiter=",",
        )
        '''

        return x_data_fall, y_data_fall, x_data_adl, y_data_adl


     # TODO: Update this to dynaically take any list of inputs, irregardless of length 
    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, data, labels, window):
            "Initialization"                
            self.data = data 
            self.labels = labels
            self.window = window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data[0])

        def __getitem__(self, index):
            #defining a function at start 
            
            def create_window(ind_vid):
                X_list =[]
                Y_list = [] 
                for i in range(0, len(ind_vid) - self.window):
                    # select the current window of the video
                    x = ind_vid[i : i + self.window]
                    y = ind_label[i : i + self.window]
                    # add the current window the list of windows
                    X_list.append(x)
                    Y_list.append(y)
                return X_list, Y_list
                
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_all_mod = []
            y_all_mod = []
            # load a single modality to work on
            for i in range(len(self.data)): 
                current_mod = self.data[i]
                current_label = self.labels[i]
                # load a single video to create windows of the data 
                ind_vid = current_mod[index]
                ind_label = current_label[index]     
                X_list, Y_list = create_window(ind_vid)
                X = np.vstack(X_list)
                X = np.reshape(X_list, (len(ind_vid) - self.window, self.window, 64, 64))
                y = np.vstack(Y_list).T
                X_all_mod.append(X)
                y_all_mod.append(y)
                
                # X should be (window-length, 64, 64, # of windows w/in video) array
                # ex. (8, 64, 64, 192) for a 200 frame video and window size of 8
                # y is array (8, # of windows w/in video)
                # this is all stored in a list with each element being a different modality
            return X_all_mod, y_all_mod


    y_data_fall = []
    x_data_fall = []
    y_data_adl = []
    x_data_adl = []

    print(dset[0])
    print(dset[1])
    print(name[0])
    print(name[1])
    for i in range(len(dset)):
        path = "H5Data\data_set-{}-imgdim64x64.h5".format(name[i]) 
        print(path)
        x1, y1, x2, y2 = load_data_set(dset[i], name[i], path)
        # where x1 and y1 contain the data and labels for videos with falls 
        # and that x2 and y2 have videos with ADL activities for training 
        x_data_fall.append(x1)
        y_data_fall.append(y1)
        x_data_adl.append(x2)
        y_data_adl.append(y2)
        # print(i)


    Test_Dataset = Dataset(
        x_data_fall, y_data_fall, window=window_len
    )
    test_dataloader = data.DataLoader(Test_Dataset, batch_size=1)


    Train_Dataset = Dataset(
        x_data_adl, y_data_adl, window=window_len
    )
    train_dataloader = data.DataLoader(Train_Dataset, batch_size=1)

    return(Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)