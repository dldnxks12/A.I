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



#### model.eval() ?

    .eval()은 'out'이라는 1개의 key를 가지고 있다. 따라서 해당 model의 출력을 얻으려면 'out' key값을 통해 얻어내야한다.
    
        
        ex) test_model = model.eval()
        
            result = test_model(x)['out']
            
            
#### torch.argmax(input) -> LongTensor

    torch.argmax()는 입력 배열의 최대값이 위치한 'index'를 반환
    
    
#### Datadownload with 'wget'

    url = "imageURL"
    wget.download(url)


