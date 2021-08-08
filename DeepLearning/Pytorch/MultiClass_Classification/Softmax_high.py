# High level Softmax cost function

'''

1. 기존 Low Lovel cost 

cost = (y_onehot * - torch.log(hypothesis)).sum(dim = 1).mean() 

2. 조금 더 나아간 cost --- F.softmax() + torch.log()

cost = (y_onehot * F.log_softmax(z, dim = 1).sum(dim = 1).mean())

3. 발전된 cost

cost = F.nll_loss(F.log_softmax(z, dim = 1), y)

    --- F.nll_loss의 인자로 y값을 one-hot 인코딩할 필요없이 그대로 넣어준다. 
    --- F.nll_loss는 F.log_softmax를 수행한 후 남은 수식들을 수행한다. 

4. High Level cost

cost = F.cross_entropy(z, y)

    --- F.cross_entropy API는 내부적으로 F.log_softmax 와 F.nll_loss()를 수행한다. 
    
    --- nll_loss와 마찬가지로 onehot 인코딩할 필요도 없다 
'''

import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand((3,5), requires_grad=True) # 3 x 5 random value tensor define
y = torch.randint(5, (3,)).long()  # long type으로 0~5까지 수 중에서 3개 (3,) size로 

cost = F.cross_entropy(z, y)
print(cost)
