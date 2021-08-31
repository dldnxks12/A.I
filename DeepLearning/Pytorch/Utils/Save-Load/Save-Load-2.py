# 이전에 저장해두었던 model과 optimizer의 가중치를 불러와서 학습하는 부분 코드 -- 잘 활욯하기

import torch
from Save_Load import CNN

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def main():

    # load한 데이터를 받아주기 위한 객체들 생성
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters())

    # load model, optimizer
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

if __name__ == "__main__":
    main()
    