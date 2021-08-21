#### Pytorch

[Model Code Example](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)


#### Model save

    net = CNN.to(device)
    torch.save(net.stat_dict(), "path")

#### Model load

    new_net = CNN.to(device) # 같은 형태로 
    new_net.load_stat_dict(torch.load("path"))


#### Model Class Weight Initialize Method

    
- nn layers

        linear1 = torch.nn.Linear(784, 256, bias=True)
        linear2 = torch.nn.Linear(256, 256, bias=True)
        linear3 = torch.nn.Linear(256, 10, bias=True)
        relu = torch.nn.ReLU()

- xavier initialization

        torch.nn.init.xavier_uniform_(linear1.weight)
        torch.nn.init.xavier_uniform_(linear2.weight)
        torch.nn.init.xavier_uniform_(linear3.weight)

[full Code](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_3_mnist_nn_xavier.ipynb)


