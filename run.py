from modulated_deform_conv import *
#"""
batch=16
H=64
W=64
C=512
K=512
R=3
S=3

stride=1
padding=1
dilation=1
groups=1
deformable_groups=512 # 512 #1

in_step=16384
#"""
"""
batch=16
H=32
W=32
C=32
K=32
R=3
S=3

stride=1
padding=1
dilation=1
groups=1
deformable_groups=512
in_step=16384
"""
device = torch.device('cuda')
cpudata=torch.rand(batch,C,H,W,requires_grad=True)
# data=torch.ones(batch,1,5,5,device='cuda',requires_grad=True)
data=cpudata.cuda()

model = DeformConv2d(C, K, (R,S), stride, padding, dilation, groups, deformable_groups=deformable_groups, in_step=in_step)
model = model.to(device)

#cpudata.retain_grad()
#data.retain_grad()

output = model(data)

#cpudata.retain_grad()
#data.retain_grad()
loss=output.sum()
loss.backward()

