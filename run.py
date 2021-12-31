import torch
import torch.nn as nn
import time
import fvcore
import fvcore.nn
import numpy as np
import csv

from modulated_deform_conv import *

RESULT_FILE = "batch_kernel_sweep.csv"
mode = "a" # Options: w or a
batch_list=[512]   #[1, 2, 4, 8, 16, 32, 64, 128, 256]
H_list=[64]
#W_list=[64]
C_list=[512]
K_list=[512]
R_list=[1,2,3,4,5,6,7,8,9,10,11]
#S_list=[3]

stride=1
padding=1
dilation=1
groups=1
deformable_groups=1 # 512 #1
KG=1 # 512 #1
in_step=16384

end_runs = False
if mode == "w":
    metrics = [["Batch", "H", "W", "C", "K", "R", "S", "", "Max mem. (MB)", "Forward time (s)", "Backward time (s)", "Act. count", "Param Count", "Total FLOPs", "Offset FLOPs"]]
else:
    metrics = []
for batch in batch_list:
    for H in H_list:
        W = H
        for C in C_list:
            for K in K_list:
                for R in R_list:
                    S = R

                    if end_runs:
                        break
                    try:
                        print("Running batch, H, W, C, K, R, S =", [batch, H, W, C, K, R, S])
                        device = torch.device('cuda')
                        cpudata=torch.rand(batch,C,H,W,requires_grad=True)
                        data=cpudata.cuda()
                        
                        #model = DeformConv2d(C, K, (R,S), stride, padding, dilation, groups, deformable_groups=deformable_groups, KG=KG, in_step=in_step)
                        #model = FusedDeformConv2d(C, K, (R,S), stride, padding, dilation, groups, deformable_groups=deformable_groups, KG=KG, in_step=in_step)
                        #model = DeformConv2dPack(C, K, (R,S), stride, padding, dilation, groups, deformable_groups=deformable_groups, KG=KG, in_step=in_step)
                        model = nn.Conv2d(C, K, (R,S), stride, padding, dilation, groups, bias=True)
                        model = model.to(device)
                        
                        warmup_output = model(data)
                        del warmup_output
                        torch.cuda.reset_peak_memory_stats()
                        
                        start_time = time.perf_counter()
                        output = model(data)
                        torch.cuda.synchronize()
                        forward_time = time.perf_counter()
                        loss=output.sum()
                        loss.backward()
                        torch.cuda.synchronize()
                        backward_time = time.perf_counter() - forward_time
                        forward_time = forward_time - start_time
                        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                        
                        act_count_module = fvcore.nn.ActivationCountAnalysis(model, (data,))
                        flop_count_module = fvcore.nn.FlopCountAnalysis(model, (data,))
                        act_count = act_count_module.total()
                        param_count = fvcore.nn.parameter_count(model)[""]
                        flop_count = None
                        offset_flop_count = flop_count_module.by_module()["conv_offset"]
                        if offset_flop_count == 0:
                            flop_count = flop_count_module.total()
                        
                        #print("Max memory usage: {} MB".format(max_mem_mb))
                        #print("Forward time {} s, backward time {} s".format(forward_time, backward_time))
                        if flop_count == None:
                            metrics.append([batch, H, W, C, K, R, S, "", max_mem_mb, forward_time, backward_time, act_count, param_count, 0, offset_flop_count])
                            #print("Activation count {}, Param count {}, Offset conv FLOP count (must be added to standard CONV total FLOP) {}".format(act_count, param_count, offset_flop_count))
                        else:
                            metrics.append([batch, H, W, C, K, R, S, "", max_mem_mb, forward_time, backward_time, act_count, param_count, flop_count, 0])
                            #print("Activation count {}, Param count {}, FLOP count {}".format(act_count, param_count, flop_count))

                    except:
                        print("Data point failed. Ending run and saving")
                        end_runs = True
                        continue
                    del cpudata
                    del data
                    del model
                    del output
                    del loss
                    del act_count_module
                    del flop_count_module
                    torch.cuda.empty_cache()
                    cur_mem_mb = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                    #print("Leftover memory after freeing: {} MB".format(cur_mem_mb))
                    assert cur_mem_mb == 0

print(metrics)
print(np.array(metrics).shape)
with open(RESULT_FILE, mode) as f:
    csvWriter = csv.writer(f, delimiter=',')
    csvWriter.writerows(metrics)
