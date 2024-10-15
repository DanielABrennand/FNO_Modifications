#%% Imports
from simvue import Run
import numpy as np
import torch
import os
import functools
import operator
import h5py
import timeit
import time

#%% Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configuration = {"Case": 'RBB Camera', #Specifying the Camera setup
                 "Pipeline": 'Sequential', #Shot-Agnostic RNN windowed data pipeline. 
                 "Calibration": 'Calcam', #CAD inspired Geometry setup
                 "Epochs": 500, 
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No', #Layerwise Normalisation
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'No', #Normalising the Variable 
                 "T_in": 10, #Input time steps
                 "T_out": 10, #Max simulation time
                 "Step": 10, #Time steps output in each forward call
                 "Modes":16, #Number of Fourier Modes,
                 "Filament Modes":40, #Number of Fourier Modes for the filament model,
                 "Width": 40, #Features of the Convolutional Kernel
                 "Loss Function": 'LP-Loss', #Choice of Loss Fucnction
                 "Resolution":1,
                 "Device": str(DEVICE),
                 "Train Percentage":0.8 #Percentage of data used for testing
                 }
#%% Data Locations
#This should just be a single file (Shot) for testing
DATA_FILE_PATH = "/home/ir-bren1/Code/Pipeline_Temp/rbb30350.h5"

OUTPUT_FOLDER_PATH = "/home/ir-bren1/Code/Pipeline_Temp"

# %% Normalisationfunctions

class UnitGaussianNormalizer(object):
    # normalization, pointwise gaussian
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class GaussianNormalizer(object):
    # normalization, Gaussian - across the entire dataset
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class RangeNormalizer(object):
    # normalization, scaling by range - pointwise
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x


    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

class MinMax_Normalizer(object):
    #normalization, rangewise but across the full domain 
    def __init__(self, minimum, maximum, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        #mymin = torch.min(x)
        #mymax = torch.max(x)
        self.mymin = minimum
        self.mymax = maximum

        self.a = (high - low)/(self.mymax - self.mymin)
        self.b = -self.a*self.mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

#%% Model Architecture

class SpectralConv2d(torch.nn.Module):
    #2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()  

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(torch.nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = torch.nn.Linear(T_in+2, self.width)
        # input channel is 12: the solution of the previous T_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)


        self.w0 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w4 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w5 = torch.nn.Conv2d(self.width, self.width, 1)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = torch.nn.Identity()

        self.fc1 = torch.nn.Linear(self.width, 128)
        self.fc2 = torch.nn.Linear(128, step)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1+x2

        x1 = self.norm(self.conv4(self.norm(x)))
        x2 = self.w4(x)
        x = x1+x2

        x1 = self.norm(self.conv5(self.norm(x)))
        x2 = self.w5(x)
        x = x1+x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        #Using x and y values from the simulation discretisation 
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(x_grid, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(y_grid, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += functools.reduce(operator.mul, list(p.size()))

        return c

#%% Data Loading and pre-processing

def FilamentIdentification(Frames,frame_range):
    #Returns an array of just the filaments from the input frames by subtracting the average frame
    #Assumes frames are in the form [X,Y,T] and that the return will be smaller in the T dimension by 
    #2 times the frame_range

    #This just makes ure its a numpy array
    Frames = np.array(Frames)

    num_frames = Frames.shape[2]
    num_smoothed_frames = num_frames-(2*frame_range)
    subtracted_frames = np.zeros((Frames.shape[0],Frames.shape[1],num_smoothed_frames))
    for n in range(num_smoothed_frames):
        frm = n+frame_range
        subset_frames = Frames[:,:,(frm-frame_range):(frm+frame_range+1)]

        avg_frame = np.mean(subset_frames,2)
        subtracted_frames[:,:,n] = Frames[:,:,frm] - avg_frame

    return subtracted_frames

T_In = configuration["T_in"]
T_Out = configuration["T_out"]

Data_File = h5py.File(DATA_FILE_PATH)
Num_Frames = len([x for x in Data_File.keys() if "frame" in x])
Shot_Frames = []
for Frame in range(Num_Frames):
        Shot_Frames.append(torch.from_numpy(np.array(Data_File["frame{}".format(str(Frame).zfill(4))]).astype(np.float32)))
#Shot_Frames is now [X,Y,T] for all T
Shot_Frames = torch.stack(Shot_Frames,-1)

Fil_Frames = torch.from_numpy(FilamentIdentification(Shot_Frames,5).astype(np.float32))[:,:,5:-5]
#Remove the first and last few frames
#NOTE replace magic number 5 shold again be in config as like cutoffvalue or something
Shot_Frames = Shot_Frames[:,:,5:-5]
#Now split into time sets:
Sets_Num = int(Shot_Frames.shape[2] - T_Out)


Input_Sets_List = []
for Set in range(Sets_Num):
    Base_Set = Shot_Frames[:,:,Set:(Set+T_In)]      
    #Collecting inputs
    Input_Sets_List.append(Base_Set)

Input_Sets_Tensor = torch.stack(Input_Sets_List) #Now in form [Index,X,Y,T]

#Normalising
#All normalisations preserve data structure
norm_strategy = configuration['Normalisation Strategy']


if norm_strategy == 'Min-Max':
    base_input_normalizer = MinMax_Normalizer(0,255)
    fil_output_normalizer = MinMax_Normalizer(-50,50)

if norm_strategy == 'Range':
    base_input_normalizer = RangeNormalizer(Input_Sets_Tensor)
    fil_output_normalizer = RangeNormalizer(Fil_Frames)

if norm_strategy == 'Gaussian':
    base_input_normalizer = GaussianNormalizer(Input_Sets_Tensor)
    fil_output_normalizer = GaussianNormalizer(Fil_Frames)

Input_Sets_Tensor = base_input_normalizer.encode(Input_Sets_Tensor)

batch_size = configuration['Batch Size']


print(Fil_Frames.shape)

#Make this section better
Input_Sets_Tensor = fil_output_normalizer.encode(Fil_Frames)
Layered_Output= []

for x in range(Fil_Frames.shape[-1]-10):
    Layered_Output.append(Input_Sets_Tensor[:,:,x:x+10])

Input_Sets_Tensor = torch.stack(Layered_Output)


#...........................

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Input_Sets_Tensor, Input_Sets_Tensor), batch_size=batch_size, shuffle=False)


#%% Model setup
#Extracting hyperparameters from the config dict

basemodes = configuration['Modes']
filmodes = configuration['Filament Modes']
width = configuration['Width']
step = configuration['Step']
T_in = configuration['T_in']
T_Out = configuration['T_out']

# Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
# pad the location (x,y)
x_grid = np.linspace(-1.5, 1.5, 448)
y_grid = np.linspace(-2.0, 2.0, 640)

#Instantiating the Models. 

#NOTE CHANGED BASE TO 16 Width
basemodel = FNO2d(basemodes, basemodes, 16)
filamentmodel = FNO2d(filmodes, filmodes, width)

basemodel.to(DEVICE)
filamentmodel.to(DEVICE)

#Loading state dictionary
if DEVICE == torch.device('cpu'):
    basemodel.load_state_dict(torch.load("/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Heuristics/FNO_Boost/Test5-BaseModel.pth",map_location=torch.device('cpu')))
    filamentmodel.load_state_dict(torch.load("/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Heuristics/FNO_Boost/Fil_Input-40W_40M-FilamentModel.pth",map_location=torch.device('cpu')))
else:
    basemodel.load_state_dict(torch.load("/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Heuristics/FNO_Boost/Test5-BaseModel.pth"))
    filamentmodel.load_state_dict(torch.load("/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Heuristics/FNO_Boost/Fil_Input-40W_40M-FilamentModel.pth"))

basemodel.eval()
filamentmodel.eval()

Base_Pred_List =[]
Fil_Pred_List = []


t1 = time.process_time()

with torch.no_grad():
    for xx, yy in test_loader:
                
        xx = xx.to(DEVICE)

        #base_pred = basemodel(xx).cpu() #Predictions of the shape [Batch,X,Y,T]
        fil_pred = filamentmodel(xx).cpu()

        #Base_Pred_List.append(base_pred[:,:,:])
        Fil_Pred_List.append(fil_pred[:,:,:])

t2 = time.process_time()

res = t2-t1

print("Time: {}".format(res))

#Base_Pred_Shot = torch.cat(Base_Pred_List) #Still of the shape [Batch,X,Y,T]
Fil_Pred_Shot = torch.cat(Fil_Pred_List)

print(Fil_Pred_Shot.shape)
#Fil_Pred_Shot = torch.squeeze(Fil_Pred_Shot[:,:,:,0]).permute(1,2,0)

#Base_Pred_Shot = base_input_normalizer.decode(Base_Pred_Shot)
Fil_Pred_Shot = fil_output_normalizer.decode(Fil_Pred_Shot)

#Base_Pred_Shot = np.squeeze(np.array(Base_Pred_Shot)[:,:,:,0])
Fil_Pred_Shot = np.squeeze(np.array(Fil_Pred_Shot)[:,:,:,0])


np.save(os.path.join(OUTPUT_FOLDER_PATH,"RBB30350_BASE_PRED.npy"),Base_Pred_Shot)
np.save(os.path.join(OUTPUT_FOLDER_PATH,"RBB30350_FIL_PRED.npy"),Fil_Pred_Shot)
