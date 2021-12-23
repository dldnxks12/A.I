#### Pytorch

[Model Code Example](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)

<br>

#### nn.Sequential vs nn.Modulelist


- nn.Squential

      입력값이 1개 일 때. 즉, 각 layer를 데이터가 순차적으로 지나갈 때 사용하면 좋은 방법

- nn.Modulelist

      nn.Module을 list로 정리하는 방법

      각 layer를 list에 전달하고 layer의 iterator를 만든다. (list에서 layer를 하나씩 받아서 처리)
    
      덕분에 forward 처리가 간단해진다는 장점이 있다. 

<br>

#### Model save

    net = CNN.to(device)
    torch.save(net.stat_dict(), "path")

#### Model load

    new_net = CNN.to(device) # 같은 형태로 
    new_net.load_stat_dict(torch.load("path"))

<br>

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

<br>

#### model.eval() ?

    .eval()은 'out'이라는 1개의 key를 가지고 있다. 따라서 해당 model의 출력을 얻으려면 'out' key값을 통해 얻어내야한다.
    
        
        ex) test_model = model.eval()
        
            result = test_model(x)['out']

<br>            
            
#### torch.argmax(input) -> LongTensor

    torch.argmax()는 입력 배열의 최대값이 위치한 'index'를 반환
    
<br>
    
#### Datadownload with 'wget'

    url = "imageURL"
    wget.download(url)


