import json
import os
from pathlib import Path
import torch
import torch.utils.data as data
def get_class_labels(path):
 filename = os.path.basename(path)
 label = filename[4]
 return label
class VideoDataset(data.Dataset):
 def __init__(self,
 root_path,
 subset,
 proc_dep = False,
 proc_rgb = True,
 spatial_transform=None,
 temporal_transform=None,
 target_transform=None,
 target_type='label'):
 self.proc_dep = proc_dep
 self.proc_rgb = proc_rgb
 self.dataRGB, self.dataDEP = self.__make_dataset(
 root_path, subset)
 self.spatial_transform = spatial_transform
 self.temporal_transform = temporal_transform
 self.target_transform = target_transform
 #self.proc_dep = proc_dep
 #self.proc_rgb = proc_rgb
 self.target_type = target_type
 def __make_dataset(self, root_path, subset):
 datasetRGB = []
 datasetDEP = []
 for name in os.listdir(root_path):
 # todo implement file verification
 if(True):
 frames = torch.load(root_path+'/'+name)
 # question about unsqueeze here and implications on batch
 frames = torch.stack(frames)
 #frames = frames.unsqueeze(1)
 if(frames.size(dim=1)!= 3):
 #print(frames.size())
 frames = torch.permute(frames,(0,3,1,2))
 #print(frames.size())
 if self.proc_dep == True:
 frames = preproc(frames,'dep')
 #print(frames.size())
 #frames = torch.permute(frames,(1,0,2,3))
 #print(frames.size())
 else:
 #print(frames.size())
 if self.proc_rgb == True:
 frames = preproc(frames,'rgb')
 #print(frames.size())
 #frames = torch.permute(frames,(1,0,2,3))
 #print("RGB")
 #print(frames.size())
 video_id = name[6:19]
 label_id = name[4]
 #print(label_id)
 rgb_or_dep = name[0:3]
 #print(name)
 videos = {
 'frames' : frames,
 'video_id' : video_id,
 'label' : label_id,
 'type' : rgb_or_dep
 }
 if(videos['type'] == 'rgb'):
 datasetRGB.append(videos)
 elif(videos['type'] == 'dep'):
 datasetDEP.append(videos)
 else:
 print('ERROR FILENAME IS BAD: ',name)
 return datasetRGB,datasetDEP
 def __len__(self):
 return len(self.data)

def get_training_data(video_path,
 dataset_name,
 subset,
 proc_dep,
 proc_rgb,
 input_type,
 file_type,
 spatial_transform=None,
 temporal_transform=None,
 target_transform=None):
 assert dataset_name in [
 'numbers','lowercase_letters'
 ]
 assert input_type in ['rgb', 'dep','both']
 assert file_type in ['zip']
 video_dataset = VideoDataset(video_path,
 subset,#'training'
 proc_dep,
 proc_rgb,
 spatial_transform=spatial_transform,
 temporal_transform=temporal_transform,
 target_transform=target_transform
 )
 training_dataRGB = video_dataset.dataRGB
 training_dataDEP = video_dataset.dataDEP
 return training_dataRGB,training_dataDEP
def get_validation_data(video_path,
 dataset_name,
 subset,
 proc_dep,
 proc_rgb,
 input_type,
 file_type,
 spatial_transform=None,
 temporal_transform=None,
 target_transform=None):

 assert dataset_name in [
 'numbers','lowercase_letters'
 ]
 assert input_type in ['rgb', 'dep','both']
 assert file_type in ['zip']
 video_dataset = VideoDataset(video_path,
 subset,
 #'validation',
 proc_dep,
 proc_rgb,
 spatial_transform=spatial_transform,
 temporal_transform=temporal_transform,
 target_transform=target_transform
 )
 validation_dataRGB = video_dataset.dataRGB
 validation_dataDEP = video_dataset.dataDEP
 return validation_dataRGB,validation_dataDEP
