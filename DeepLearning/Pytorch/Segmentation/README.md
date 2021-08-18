#### Segmentation Model 

    1. FCN
    2. SegNet
    3. U-Net
    4. deeplabv3 in torchvision


#### model.eval() ?

    .eval()은 'out'이라는 1개의 key를 가지고 있다. 따라서 해당 model의 출력을 얻으려면 'out' key값을 통해 얻어내야한다.
    
        
        ex) test_model = model.eval()
        
            result = test_model(x)['out']
            
            
#### torch.argmax(input) -> LongTensor

    torch.argmax()는 입력 배열의 최대값이 위치한 'index'를 반환한다!
    
#### wget으로 데이터 다운로드 

    url = "imageURL"
    wget.download(url)

   
