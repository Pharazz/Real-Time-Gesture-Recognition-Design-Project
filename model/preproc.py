from numpy.lib.function_base import append
### PREPROC FUNCTIONS
import cv2 as cv2
import numpy as np
import torch
from google.colab.patches import cv2_imshow
import time
def preproc(tensorFile,typeOfTensor):
 import os
 rgbs = []
 deps = []
 rgbKNN = []
 depGauss = []
 torch.set_default_tensor_type(torch.FloatTensor)
 if typeOfTensor == 'rgb':
 rgbKNN = []
 #setup for background subtraction
 backSubRGB = cv2.createBackgroundSubtractorKNN()
 ################################
 #PRE-PROCESSING TECHNIQUES
 ################################
 for i in range (0,30):
 tensor_image = tensorFile[i]
 tensor_image = tensor_image.permute(1,2,0)
 tensor_image = torch.squeeze(tensor_image)
 temp = np.floor(tensor_image*256)
 tensor_image = torch.clamp(temp, 0, 255)
 tensor_image = tensor_image.numpy()
 tensor_image = tensor_image.astype(np.uint8)
 
 #background subtraction
 fgMaskRGB1 = backSubRGB.apply(tensor_image)
 #saving RGB preprocessed data
 #first 4 frames of motion based background subtraction not used
 if(i>3):
 rgbKNN.append(fgMaskRGB1[:] )
 for i in range(0,len(rgbKNN)):
 rgbKNN[i] = torch.FloatTensor(rgbKNN[i]/255)

 rgbKNN = torch.stack(rgbKNN)
 rgbKNN = torch.unsqueeze(rgbKNN,0)
 return(rgbKNN)
 elif typeOfTensor == 'dep':
 depGauss = []
 for i in range(0,30):
 tensor_image = tensorFile[i]
 tensor_image = torch.squeeze(tensor_image)
 temp = np.floor(tensor_image*256)
 tensor_image = torch.clamp(temp, 0, 255)
 tensor_image = tensor_image.numpy()
 tensor_image = tensor_image.astype(np.uint8)
 median = cv2.medianBlur(tensor_image,5)
 tensor_image3 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv2.THRESH_BINARY,11,2)
 depGauss.append(tensor_image3[:])

 for i in range(0,len(depGauss)):
 depGauss[i] = torch.FloatTensor(depGauss[i]/255)
 depGauss_tens = torch.stack(depGauss)
 depGauss_tens = torch.unsqueeze(depGauss_tens,0)
 return depGauss_tens 
