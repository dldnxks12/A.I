import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from UNET import UNET


# hyperparameters

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 3
TRAIN_IMG_DIR = "C:/Users/USER/Desktop/Carvana/train/"
TRAIN_MASK_DIR = "C:/Users/USER/Desktop/Carvana/train_mask/"
VAL_IMG_DIR = "C:/Users/USER/Desktop/Carvana/val/"
VAL_MASK_DIR = "C:/Users/USER/Desktop/Carvana/val_mask/"


def main():

    model = UNET(in_channels= 3, out_channels= 1).to(device = DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # UNET Class의 마지막 layer에서 sigmoid 해준다면 그냥 nn.BCELoss 쓰고, nn.BCEWithLogitsLoss는 내부적으로 Sigmoid 수행해 줌
    '''
        out_channels가 1개 이상이면 loss_fn = Cross Entropy Loss 사용
    '''
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_dataset = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = CarvanaDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    train_loader =  DataLoader(dataset = train_dataset, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    val_loader =  DataLoader(dataset = val_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)


    for epoch in range(NUM_EPOCHS):
        avg_loss = 0
        batch_length = len(train_loader)
        for data, targets in train_loader:
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)  # channel dimension 처리

            predictions = model(data)
            loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss / batch_length

            print("loss : ", loss)
        print("Average Loss : ", avg_loss)

    print("#--------- train finished ------ #")

    print("save Model")

    state = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }

    torch.save(state, "./model.pth.tar")

    print("#--------- Valid Start ------ #")

    check_acc(val_loader, model)

    print("#--------- Valid finished ------ #")

def check_acc(val_loader, model, device = 'cude'):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device = DEVICE)
            y = y.to(device = DEVICE).unsqueeze(1)

            preds = model(x)
            preds = torch.sigmoid(preds)

            preds = ( preds > 0.5 ).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8 )

    print(f"ACC : {num_correct/num_pixels}")
    print(f"Dice Score : {dice_score/len(val_loader)}")


if __name__ == "__main__":
    main()



