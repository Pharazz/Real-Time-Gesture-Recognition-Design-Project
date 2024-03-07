import traceback
sample_size = 1
sample_durationRGB = 26
sample_durationDEP = 30
dirTrain = '/content/drive/Shareddrives/Capstone/Training'
dirValid = '/content/drive/Shareddrives/Capstone/Validation'
labeller = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
training_dataRGB, _ = get_training_data(dirTrain,'numbers',0,True,True,'both','zip')
validation_dataRGB,_ = get_validation_data(dirValid,'numbers',0,True,True,'both','zip')
########################################################################################################
videoFrameDEP = [(video['frames'] ,labeller.copy()) for video in training_dataDEP]
for i,data in enumerate(training_dataDEP):
 videoFrameDEP[i][1][int(data['label'])] = 1.0
videoFrameDEP = [(video[0].to('cuda') ,torch.tensor(video[1]).to('cuda')) for video in videoFrameDEP]
train_loaderDEP = torch.utils.data.DataLoader(dataset=videoFrameDEP,batch_size=4,shuffle=True)
videoFrameRGB = [(video['frames'] ,labeller.copy()) for video in training_dataRGB]
for i,data in enumerate(training_dataRGB):
 videoFrameRGB[i][1][int(data['label'])] = 1.0
videoFrameRGB = [(video[0].to('cuda') ,torch.tensor(video[1]).to('cuda')) for video in videoFrameRGB]
train_loaderRGB = torch.utils.data.DataLoader(dataset=videoFrameRGB,batch_size=4,shuffle=True)
videoFrameDEPvalid = [(video['frames'] ,labeller.copy()) for video in validation_dataDEP]
for i,data in enumerate(validation_dataDEP):
 videoFrameDEPvalid[i][1][int(data['label'])] = 1.0
videoFrameDEPvalid = [(video[0].to('cuda') ,torch.tensor(video[1]).to('cuda')) for video in videoFrameDEPvalid]
valid_loaderDEP = torch.utils.data.DataLoader(dataset=videoFrameDEPvalid,batch_size=4,shuffle=True)
videoFrameRGBvalid = [(video['frames'] ,labeller.copy()) for video in validation_dataRGB]
for i,data in enumerate(validation_dataRGB):
 videoFrameRGBvalid[i][1][int(data['label'])] = 1.0
videoFrameRGBvalid = [(video[0].to('cuda') ,torch.tensor(video[1]).to('cuda')) for video in videoFrameRGBvalid]
valid_loaderRGB = torch.utils.data.DataLoader(dataset=videoFrameRGBvalid,batch_size=4,shuffle=True)
#########################################################################################################
try:
 #Load new or saved model for DEPTH
 #modelDEP = generate_model(10,n_input_channels = 1,n_classes=10, shortcut_type="A")
 modelDEP = torch.load('/content/drive/Shareddrives/Capstone/modelDEP_Adam_PreProc_Res10_lr-0p001_batch4SCRIPTABLE.pt')

 #Proceed
 modelDEP.to('cuda')
 modelDEP.train()
 lrateDEP = 0.001
 optimizerDEP = torch.optim.Adam(modelRGB.parameters(), lr=lrateDEP)
 path = "/content/drive/Shareddrives/Capstone/logs/Training_Adam_modelDEPproc_Res10_lr-0p001_batch4SCRIPTABLE_upd_part2.csv"
 rgb_train_loggerDEP = Logger(path, ['epoch','batch', 'iter','loss','acc','acc avg','lr','Guess1','Guess2','Guess3','Guess4','Label1','Label2','Label3','Label4'])
 path = "/content/drive/Shareddrives/Capstone/logs/Validation_Adam_modelDEPproc_Res10_lr-0p001_batch4SCRIPTABLE_upd_part2.csv"
 rgb_val_loggerDEP = Logger(path, ['epoch','batch','loss','acc','Guess1','Guess2','Guess3','Guess4','Label1','Label2','Label3','Label4'])
 train_and_valid_epoch( 50,
 'RGB',
 train_loaderDEP,
 valid_loaderDEP,
 modelDEP,
 torch.nn.CrossEntropyLoss().to('cuda'),
 optimizerDEP,
 torch.device('cuda'),
 lrateDEP,
 epoch_logger = None,
 batch_logger = rgb_train_loggerDEP,
 valid_logger = rgb_val_loggerDEP,
 tb_writer=None,
 distributed=False)
except:
 print("Caught an Exception in DEP")
 traceback.print_exc()
finally:
 torch.save(modelRGB,'/content/drive/Shareddrives/Capstone/modelDEP_Adam_PreProc_Res10_lr-0p001_batch4SCRIPTABLE_part2.pt')
print('\n\n\n\n\n\n\n')
print('Avg Loss '+str(depVal))
print('Avg Loss '+str(RGBval))
#########################################################################################################
try:
 #Load new or saved model for RGB
 #modelRGB = generate_model(10,n_input_channels = 1,n_classes=10, shortcut_type="A")
 modelRGB = torch.load('/content/drive/Shareddrives/Capstone/modelRGB_Adam_PreProc_Res10_lr-0p001_batch4SCRIPTABLE.pt')

 #Proceed
 modelRGB.to('cuda')
 modelRGB.train()
 lrateRGB = 0.001
 optimizerRGB = torch.optim.Adam(modelRGB.parameters(), lr=lrateRGB)
 path = "/content/drive/Shareddrives/Capstone/logs/Training_Adam_modelRGBproc_Res34_lr-0p001_batch4SCRIPTABLE_upd_part2.csv"
 rgb_train_logger = Logger(path, ['epoch','batch', 'iter','loss','acc','acc avg','lr','Guess1','Guess2','Guess3','Guess4','Label1','Label2','Label3','Label4'])
 path = "/content/drive/Shareddrives/Capstone/logs/Validation_Adam_modelRGBproc_Res10_lr-0p001_batch4SCRIPTABLE_upd_part2.csv"
 rgb_val_logger = Logger(path, ['epoch','batch','loss','acc','Guess1','Guess2','Guess3','Guess4','Label1','Label2','Label3','Label4'])
 train_and_valid_epoch( 50,
 'RGB',
 train_loaderRGB,
 valid_loaderRGB,
 modelRGB,
 torch.nn.CrossEntropyLoss().to('cuda'),
 optimizerRGB,
 torch.device('cuda'),
 lrateRGB,
 epoch_logger = None,
 batch_logger = rgb_train_logger,
 valid_logger = rgb_val_logger,
 tb_writer=None,
 distributed=False)
except:
 print("Caught an Exception in RGB")
 traceback.print_exc()
Research on Vision-Based Gesture Recognition
82
Jamil, Nassernia, Shah, and Sinclair
finally:
 torch.save(modelRGB,'/content/drive/Shareddrives/Capstone/modelRGB_Adam_PreProc_Res10_lr-0p001_batch4SCRIPTABLE_part2.pt')
