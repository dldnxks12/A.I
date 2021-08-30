import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from UNET import UNET


# hyperparameters

LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 3
TRAIN_IMG_DIR = "C:/Users/USER/PycharmProjects/A.I/Project2/U-Net/BUSI2/train/"
TRAIN_MASK_DIR = "C:/Users/USER/PycharmProjects/A.I/Project2/U-Net/BUSI2/train-mask/"
VAL_IMG_DIR = "C:/Users/USER/PycharmProjects/A.I/Project2/U-Net/BUSI2/test/"
VAL_MASK_DIR = "C:/Users/USER/PycharmProjects/A.I/Project2/U-Net/BUSI2/test-mask/"


def main():

    model = UNET(in_channels= 3, out_channels= 1).to(device = DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # UNET Class의 마지막 layer에서 sigmoid 해준다면 그냥 nn.BCELoss 쓰고, nn.BCEWithLogitsLoss는 내부적으로 Sigmoid 수행해 줌
    '''
        out_channels가 1개 이상이면 loss_fn = Cross Entropy Loss 사용
    '''
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_dataset = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 1)
    train_dataset2 = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 2)
    train_dataset3 = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 3)
    train_dataset4 = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype=4)

    train_loader =  DataLoader(dataset = train_dataset, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    train_loader2 =  DataLoader(dataset = train_dataset2, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    train_loader3 =  DataLoader(dataset = train_dataset3, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    train_loader4 = DataLoader(dataset=train_dataset4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_dataset = CarvanaDataset(VAL_IMG_DIR, VAL_MASK_DIR, TFtype = 1)
    val_loader =  DataLoader(dataset = val_dataset, batch_size= 3, shuffle=True, drop_last=True)

    T_loader = []
    T_loader.append(train_loader)
    T_loader.append(train_loader2)
    T_loader.append(train_loader3)
    T_loader.append(train_loader4)

    model.train()
    print("#--------- train Start ------ #")

    for epoch in range(NUM_EPOCHS):
        idx = 0
        avg_loss = 0.0
        batch_length = len(train_loader)
        for loader_idx, loader in enumerate(T_loader):
            for data, targets in loader:
                idx += 1
                data = data.to(device=DEVICE)
                targets = targets.float().to(device=DEVICE)  # channel dimension 처리

                predictions = model(data)
                loss = loss_fn(predictions, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / batch_length

                if idx % 10 == 0:
                    print("loss : ", loss.item())
            print(f" Loader {loader_idx} Epoch {epoch} Average Loss {avg_loss} ")

    print("#--------- train finished ------ #")

    print("save Model")

    state = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }

    torch.save(state, "model.pth.tar")

    print("#--------- Valid Start ------ #")

    check_acc(val_loader, model)

    print("#--------- Valid finished ------ #")

def check_acc(val_loader, model, device = 'cude'):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        print("length of Val_loader : ", len(val_loader))
        for x, y in val_loader:
            x = x.to(device = DEVICE)
            y = y.to(device = DEVICE).unsqueeze(1)

            print("x.shape", x.shape)
            print("y.shape", y.shape)
            preds = torch.sigmoid(model(x)) # get hypothesis and do activation function
            print("pred.shape", preds)

            preds = ( preds > 0.5 ).float() # 0.5 이상인 Pixel == True or 1

            num_correct += (preds == y).sum() # True == 1 , False == 0
            num_pixels  += torch.numel(preds) # torch.numel : number of element
            print("num_correct", num_correct)
            print("num_pixels", num_pixels)

            dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8 ) # IoU Score

    print(f" Got {num_correct} / {num_pixels} with ACC {num_correct/num_pixels}")
    print(f"Dice Score : {dice_score/len(val_loader)}")


if __name__ == "__main__":
    main()



