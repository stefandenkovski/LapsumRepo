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
            "results/Edits/{}/night_time_{}.csv".format(dset[0] + set[-1:], start_time),
            x_info_fall_ToD,
            delimiter=",",
        )

        return x_data_fall, y_data_fall, x_data_adl, y_data_adl


     # TODO: Update this to dynaically take any list of inputs, irregardless of length 
    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, data, labels, window):
            "Initialization"                
            self.data
            self.labels
            self.window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data[0])

        def __getitem__(self, index):
            #defining a function at start 
            
            def create_window(ind_vid):
                X_list =[]
                Y_list = [] 
                for i in range(0, len(ind_vid1) - self.window):
                    # select the current window of the video
                    x = ind_vid[i : i + self.window]
                    y = ind_label[i : i + self.window]
                    # add the current window the list of windows
                    X_list.append(x)
                    Y_list.append(y)
                    return(X_list, Y_list)
                
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_all_mod = []
            y_all_mod = []
            # load a single modality to work on
            for i in range(len(self.data)): 
                current_mod = self.data[i]
                current_label = self.label[i]
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
        path = "processed_data\data_set-{}-imgdim64x64.h5".format(name[i])
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


'''

# Old code
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
            "results/Edits/{}/night_time_{}.csv".format("ONI_DepthONI_IR", start_time),
            x_info_fall_ToD,
            delimiter=",",
        )

        return x_data_fall, y_data_fall, x_data_adl, y_data_adl



    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, data1, labels1, data2, labels2, window):
            "Initialization"
            self.labels1 = labels1
            self.data1 = data1
            self.labels2 = labels2
            self.data2 = data2
            self.window = window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data1)

        def __getitem__(self, index):
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_list1 = []
            Y_list1 = []
            X_list2 = []
            Y_list2 = []
            # load a single video to chop up into windows
            print(index)
            ind_vid1 = self.data1[index]
            ind_label1 = self.labels1[index]
            # loop through each frame of the video (stopping window length short)
            for i in range(0, len(ind_vid1) - self.window):
                # select the current window of the video
                X1 = ind_vid1[i : i + self.window]
                y1 = ind_label1[i : i + self.window]
                # add the current window the list of windows
                X_list1.append(X1)
                Y_list1.append(y1)
            # load a single video to chop up into windows
            ind_vid2 = self.data2[index]
            ind_label2 = self.labels2[index]
            # loop through each frame of the video (stopping window length short)
            for i in range(0, len(ind_vid2) - self.window):
                # select the current window of the video
                X2 = ind_vid2[i : i + self.window]
                y2 = ind_label2[i : i + self.window]
                # add the current window the list of windows
                X_list2.append(X2)
                Y_list2.append(y2)
            # convert lists into arrays with proper size
            X1 = np.vstack(X_list1)
            X1 = np.reshape(X_list1, (len(ind_vid1) - self.window, self.window, 64, 64))
            y1 = np.vstack(Y_list1).T

            X2 = np.vstack(X_list2)
            X2 = np.reshape(X_list2, (len(ind_vid2) - self.window, self.window, 64, 64))
            y2 = np.vstack(Y_list2).T
            # X should be (window-length, 64, 64, # of windows w/in video) array
            # ex. (8, 64, 64, 192) for a 200 frame video and window size of 8
            # y is array (8, # of windows w/in video)
            return X1, y1, X2, y2


    y_data_fall = []
    x_data_fall = []
    y_data_adl = []
    x_data_adl = []

    print(dset[0])
    print(dset[1])
    print(name[0])
    print(name[1])
    for i in range(2):
        path = "processed_data\data_set-{}-imgdim64x64.h5".format(name[i])
        x1, y1, x2, y2 = load_data_set(dset[i], name[i], path)
        x_data_fall.append(x1)
        y_data_fall.append(y1)
        x_data_adl.append(x2)
        y_data_adl.append(y2)
        # print(i)

    print("Fall lengths")
    print("Length data", len(x_data_fall[0]))
    print("Length data", len(y_data_fall[0]))
    print("Length data", len(x_data_fall[1]))
    print("Length data", len(y_data_fall[1]))
    print("ADL lengths")
    print("Length data", len(x_data_adl[0]))
    print("Length data", len(y_data_adl[0]))
    print("Length data", len(x_data_adl[1]))
    print("Length data", len(y_data_adl[1]))


    Test_Dataset = Dataset(
        x_data_fall[0], y_data_fall[0], x_data_fall[1], y_data_fall[1], window=window_len
    )
    test_dataloader = data.DataLoader(Test_Dataset, batch_size=1)


    Train_Dataset = Dataset(
        x_data_adl[0], y_data_adl[0], x_data_adl[1], y_data_adl[1], window=window_len
    )
    train_dataloader = data.DataLoader(Train_Dataset, batch_size=1)

    return(Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)
'''