import numpy as np

import torch
from torch.autograd import Variable

def to_var(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def to_np(tensor):
    return tensor.detach().data.cpu().numpy()


def init_weights(m, mode='MSRAFill'):
    import torch.nn as nn
    import torch.nn.init as init
    from torchlab.nnlib.init import XavierFill, MSRAFill
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if mode == 'GaussianFill':
            init.normal_(m.weight, std=0.001)
        elif mode == 'MSRAFill':
            MSRAFill(m.weight)            
        else:
            raise ValueError
        if m.bias is not None:
            init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        XavierFill(m.weight)
        init.constant_(m.bias, 0)

def init_with_pretrain(model, pretrained_dict):
    model_dict = model.state_dict()
    nummodel = len(model_dict)
    numpretrain = len(pretrained_dict)
    if list(pretrained_dict.keys())[0][0:7]=='module.':
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    elif list(pretrained_dict.keys())[0][0:7+6]=='model.module.':
        pretrained_dict = {k[7+6:]: v for k, v in pretrained_dict.items() if k[7+6:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    print ('update %d/%d params. from %d params.'%(len(pretrained_dict), nummodel, numpretrain))
    model.load_state_dict(model_dict)

def adjust_learning_rate(optimizer, iteration, BASE_LR=1e-4,
                         WARM_UP_FACTOR=1.0/3.0, WARM_UP_ITERS=500,
                         STEPS=[0, 60000, 80000], GAMMA=0.1):
    # do something 
    if iteration < WARM_UP_ITERS:
        alpha = float(iteration) / WARM_UP_ITERS
        lr_new = (WARM_UP_FACTOR * (1 - alpha) + alpha) * BASE_LR
    elif iteration >= WARM_UP_ITERS:
        for decay_steps_ind in range(0, len(STEPS) - 1):
            if iteration < STEPS[decay_steps_ind+1] and iteration >= STEPS[decay_steps_ind]:
                lr_new = BASE_LR * (GAMMA**decay_steps_ind)
                break
        if iteration >= STEPS[-1]:
            lr_new = BASE_LR * (GAMMA**(len(STEPS)-1))
    for i in range(len(optimizer.param_groups)):
        factor = optimizer.param_groups[i]['lr'] / optimizer.param_groups[0]['lr']
        optimizer.param_groups[i]['lr'] = factor * lr_new
    
    return optimizer.param_groups[0]['lr']


def draw_lr_schedule():
    from torch.nn import Parameter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    param1 = Parameter(torch.zeros([3, 3, 256, 256], dtype=torch.float32))
    param2 = Parameter(torch.zeros([3, 3, 256, 256], dtype=torch.float32))
    base_lr = 1e-4
    params = [{'params': [param1], 'lr': base_lr * 1.},
              {'params': [param2], 'lr': base_lr * 2., 'weight_decay': 0.}]
    #params = [param1, param2]

    optimizer = torch.optim.Adam(params, base_lr, weight_decay=0.0001)

    iterations = range(100000)
    lrs = []
    for iteration in iterations:
        lr = adjust_learning_rate(optimizer, iteration, BASE_LR=1e-4,
                                 WARM_UP_FACTOR=1.0/3.0, WARM_UP_ITERS=5000,
                                 STEPS=[0, 60000, 80000], GAMMA=0.1)
        lrs.append(lr)

    plt.figure()
    plt.plot(iterations, lrs)
    plt.savefig("lr_schedule.jpg")






