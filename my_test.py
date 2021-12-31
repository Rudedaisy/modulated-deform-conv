from modulated_deform_conv import *
#"""
batch=256
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
deformable_groups=1 # 512 #1

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
cpudata=torch.rand(batch,C,H,W,requires_grad=True)
# data=torch.ones(batch,1,5,5,device='cuda',requires_grad=True)
data=cpudata.cuda()
offset=torch.zeros(batch,2*R*S*deformable_groups,H,W,device='cuda',requires_grad=True)
mask=torch.ones(batch,R*S,H,W,device='cuda',requires_grad=True)
weight=torch.rand(K,C,R,S,device='cuda',requires_grad=True)
bias=torch.zeros(C,device='cuda',requires_grad=True)
'''
class DeformConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
'''

cpudata.retain_grad()
data.retain_grad()
offset.retain_grad()
mask.retain_grad()
weight.retain_grad()
bias.retain_grad()

#print("--Input data--")
#print(data)
out=deform_conv2d(data,offset,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
#out=modulated_deform_conv2d(data,offset,mask,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
#print("--Output data--")
#print(out)

max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
print("Max memory usage: {} MB".format(max_mem_mb))

#######exit()
cpudata.retain_grad()
data.retain_grad()
offset.retain_grad()
mask.retain_grad()
weight.retain_grad()
bias.retain_grad()

loss=out.sum()
#print("--Loss--")
#print(loss)
#print("--Data grad--")
#print(data.grad)
#print("--Offset grad--")
#print(offset.grad)
#print("--Weight grad--")
#print(weight.grad)
#print("--Bias grad--")
#print(bias.grad)
loss.backward()
#print("--Data grad after loss.backward()--")
#print(data.grad)
#print("--CPU data grad--")
#print(cpudata.grad)
#print("--Bias grad--")
#print(bias.grad)
max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
print("Max memory usage: {} MB".format(max_mem_mb))
