from os import name
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from functions import create_pytorch_dataset
from functions import get_window_metrics
from functions import get_frame_metrics
from functions import animate
import functions
from functions import animate
from functions import get_total_performance_metrics

# Lets load the H%PY dataset into a pytorch dataset class.Please see 
# dataset_creator on how to generate the H5PY file. 
window_len = 8
stride = 1
fair_comparison = False


list_of_models = ['Thermal_EditFair_RegularLoss2021-06-11-10-23-32', 'ONI_Depth_FilledFair_RegularLoss2021-06-10-14-14-34', 'ONI_IR_EditFair_RegularLoss2021-05-28-11-06-21', 'IP_EditFair_RegularLoss2021-05-27-23-44-29', 'ZED_Depth_FilledFair_RegularLoss2021-06-11-12-44-52', 'ZED_RGB_EditFair_RegularLoss2021-06-11-13-41-50'] 
list_of_datasets = ['Thermal_Edit', 'ONI_Depth_Filled', 'ONI_IR_Edit', 'IP_Edit', 'ZED_Depth_Filled', 'ZED_RGB_Filled'] 
list_of_files = ["Edits/Thermal", "Edits/ONI_Depth", "Edits/ONI_IR", "Edits/IP", "Edits/ZED_Depth", "Edits/ZED_RGB"] 

def full_pipeline(name, dset, filepath, window_len, fair_comparison, path, stride):


    Test_Dataset, test_dataloader, Train_Dataset, train_dataloader = create_pytorch_dataset(name, dset, path, window_len, fair_comparison, stride)


    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            # first layer
            self.ec1 = nn.Conv3d(1, 16, (5, 3, 3), stride=1, padding=(2, 1, 1),)
            self.em1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
            #self.ed1 = nn.Dropout3d(p=0.25)
            # second layer
            self.ec2 = nn.Conv3d(16, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
            self.em2 = nn.MaxPool3d((2, 2, 2), return_indices=True)
            #self.ed2 = nn.Dropout3d(p=0.25)
            # third layer
            self.ec3 = nn.Conv3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
            self.em3 = nn.MaxPool3d((2, 2, 2), return_indices=True)
            # encoding done, time to decode
            self.dc1 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
            self.dm1 = nn.MaxUnpool3d((2, 2, 2))
            # inverse of 2nd Conv
            self.dc2 = nn.ConvTranspose3d(8, 8, (5, 3, 3), stride=1, padding=(2, 1, 1))
            self.dm2 = nn.MaxUnpool3d((2, 2, 2))
            # inverse of 1st Conv
            self.dc3 = nn.ConvTranspose3d(8, 16, (5, 3, 3), stride=1, padding=(2, 1, 1))
            self.dm3 = nn.MaxUnpool3d((1, 2, 2))
            # final inverse
            self.dc4 = nn.ConvTranspose3d(16, 1, (5, 3, 3), stride=1, padding=(2, 1, 1))

        def forward(self, x):
            # *** start of encoder
            x = x.permute(1, 0, 2, 3, 4)  # reorder to have correct dimensions
            
            # (batch_size, chanels, depth, width, height)
            _ec1 = F.relu(self.ec1(x))
            _em1, i1 = self.em1(_ec1)


            #_em1 = self.ed1(_em1) # dropout layer
            # second layer 
            _ec2 = F.relu(self.ec2(_em1))
            _em2, i2 = self.em2(_ec2)
            
            #_em2 = self.ed2(_em2) # dropout layer
            # third layer
            _ec3 = F.relu(self.ec3(_em2))
            _em3, i3 = self.em3(_ec3)
        
            # print("====== Encoding Done =========")
            # *** encoding done, time to decode
            _dc1 = F.relu(self.dc1(_em3))
            _dm1 = self.dm1(_dc1, i3, output_size=_em2.size())
            
            # second layer
            _dc2 = F.relu(self.dc2(_dm1))
            _dm2 = self.dm2(_dc2, i2)
            
            # third layer
            _dc3 = F.relu(self.dc3(_dm2))
            _dm3 = self.dm3(_dc3, i1)
            
            re_x = torch.tanh(self.dc4(_dm3))
            
            return re_x
        

    # Now lets train our model

    # prepare for GPU training 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.cuda.empty_cache()

    # and lets set the hyperparameters! 

    dropout = 0.25
    learning_rate = 0.0002
    num_epochs = 20
    chunk_size = 128
    forward_chunk = 8 
    forward_chunk_size = 8 # this is smaller due to memory constrains 

    # select which model - you could load your own or put it in the function above 
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    def train_model(filepath):
        model.train()
        for epoch in range(num_epochs):
            val_loss = 0
            for i, (sample, labels) in enumerate(train_dataloader):
                # ===================forward=====================
                sample = sample.to(device, dtype=torch.float)
                # split sample into smaller sizes due to GPU memory constraints
                chunks = torch.split(sample, chunk_size, dim=1)
                for chunk in chunks:
                    output = model(chunk)
                    output = output.to(device).permute(1, 0, 2, 3, 4)
                    model.zero_grad()
                    loss = loss_fn(output, chunk)
                    # ===================backward====================
                    # Getting gradients w.r.t. parameters
                    loss.backward()
                    # Updating parameters
                    optimizer.step()
                    # Clear gradients w.r.t. parameters
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            # ===================log========================
            print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.item()))
            torch.save(model.state_dict(), filepath) # save the model each epoch at location filepath
            
        torch.cuda.empty_cache()
        


    def foward_pass(path):
        model.load_state_dict(torch.load(path)) # load a saved model 
        model.eval()
        frame_stats = [] 
        window_stats = [] 

        with torch.no_grad():
            print("foward pass occuring")
            # just forward pass of model on test dataset
            for j, (sample, labels) in enumerate(test_dataloader):
                print(j)
                # foward pass to get output
                torch.cuda.empty_cache()
                sample = sample.to(device, dtype=torch.float)
                chunks = torch.split(sample, forward_chunk, dim=1)
                recon_vid = []
                for chunk in chunks:
                    output = model(chunk)
                    output = output.to(device).permute(1, 0, 2, 3, 4)
                    recon_vid.append(output)
                    torch.cuda.empty_cache()

                output = torch.cat(recon_vid, dim=1)
                # convert tensors to numpy arrays for easy manipluations
                sample = sample.data.cpu().numpy()
                output = output.data.cpu().numpy()
                labels = labels.data.cpu().numpy()

                frame_mean, frame_std, frame_labels = get_frame_metrics(output, sample, labels, window_len)
                mean_window_error, std_window_error, window_labels = get_window_metrics(output, sample, labels, window_len)
                frame_stats.append([frame_mean, frame_std, frame_labels])
                window_stats.append([mean_window_error, std_window_error, window_labels])
                '''
                if j % 3 == 0:
                    animate(sample[0, :, :, :, :], output[0, :, :, :, :], frame_mean, dset, start_time)
                '''
                

        return(frame_stats, window_stats)
    
    frame_stats, window_stats = foward_pass(filepath)
    modality = (name 
    + 'Fair_'
    + 'RegularLoss')


    np.save("Recon_Errors\\frame_stats_{}.npy".format(modality), frame_stats)
    np.save("Recon_Errors\\window_stats{}.npy".format(modality), window_stats)
    frame_stats2 = np.load("Recon_Errors\\frame_stats_{}.npy".format(modality), allow_pickle=True)
    window_stats2 = np.load("Recon_Errors\\window_stats{}.npy".format(modality), allow_pickle=True)
    get_total_performance_metrics(frame_stats, window_stats, window_len)
    
    return()


for i in range(len(list_of_models)):
    modelpath = list_of_models[i]
    name = list_of_datasets[i]
    dset = list_of_files[i]
    path = "H5Data\Data_set-{}-imgdim64x64.h5".format(name)
    full_pipeline(name, dset, modelpath, window_len, fair_comparison, path, stride)