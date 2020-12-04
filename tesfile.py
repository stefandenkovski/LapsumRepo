import numpy as np
import h5py
import cv2
import torch 
import scipy.signal

print("Hello World")

x = "Hello world 2"
print(x)

loss_kernal = [
[[ 0.,  -1.,  0.], 
[ -1.,  -1.,  -1.],
[ 0.,  -1.,  0.]], 

[[ 0.,  -1.,  0.], 
[ -1.,  14.,  -1.],
[ 0.,  -1.,  0.]], 

[[ 0.,  -1.,  0.], 
[ -1.,  -1.,  -1.],
[ 0.,  -1.,  0.]] ]

loss_kernal = np.array(loss_kernal)
fake_image = np.random.rand(8,64,64)
convlved = scipy.signal.fftconvolve(fake_image, loss_kernal, mode='valid', axes=None)
adjactent_loss = sum(convlved)
print(adjactent_loss)