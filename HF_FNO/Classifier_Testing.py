import torch.nn as nn
from torch import cuda
import torch
from torch import flatten,from_numpy
import numpy as np
import h5py
#from matplotlib import pyplot as plt
import time

f = h5py.File("/home/ir-bren1/Code/Classifier_Temp/Testing_RBB/rbb30356.h5")
Frames = len(f.keys()) - 4
x = []
for n in range(Frames):
    frame = np.expand_dims(np.array(f["frame{}".format(str(n).zfill(4))]),axis=0)
    if n == 0:
        x = frame
    else:
        x = np.concatenate((x,frame))

y = from_numpy(np.float32(x[:,:,:])).unsqueeze(0)
print(y.shape)

DEVICE = "cuda" if cuda.is_available() else "cpu"

class AlexNet(nn.Module):
    def __init__(self,dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x,1)
        x = self.classifier(x)
        return x
    
net = AlexNet()
net.load_state_dict(torch.load("/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Classifier/Heuristics/LHClassifierNewBuild_quadratic-coda_Full.pth",map_location=torch.device('cpu')))
net.eval()

outs = np.zeros(Frames)

t1 = time.process_time()
with torch.no_grad():
    for n in range(Frames):
        outs[n] = net(y[0,n,:,:].unsqueeze(0).unsqueeze(0))
t2 = time.process_time()

res = t2-t1

print("Time: {}".format(res))


print(outs.shape)
#plt.plot(outs)
#plt.savefig("/home/ir-bren1/Code/Classifier_Temp/ClassRun.png")
