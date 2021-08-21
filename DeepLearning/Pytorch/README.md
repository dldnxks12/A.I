#### Pytorch

[Model Code Example](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)


#### Model save

    net = CNN.to(device)
    torch.save(net.stat_dict(), "path")

#### Model load

    new_net = CNN.to(device) # 같은 형태로 
    new_net.load_stat_dict(torch.load("path"))


#### Model Class Weight Initialize Method

    initialize Code 참고 
