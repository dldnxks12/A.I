import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # bias false for batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # bias false for batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x) # Go Double Conv

class UNET(nn.Module):
    # out_channel : category ... (1 -> binary segmentation)
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()

        # ModuleList : 각 layer를 list에 전달하고, layer를 하나씩 던져주는 Iterator 생성
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Down Part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up Par t of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size = 2 , stride = 2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # 1x1 Conv for change just channel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse

        # (Up -> DoubleConv)를 하나의 세트로 하기 위해 아래와 같이..
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # ConvTranspose2d (Upsampling)
            skip_connection = skip_connections[idx//2]

            # MaxPooling할 때 2로 나누어지지 않는다면, floor 연산을 수행한다.
            # 따라서 Upsample해서 Concatenate할 때, 사이즈가 안맞는 경우가 생긴다.
            # 이를 방지하기 위해 다음과 같은 처리를 추가한다.
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:]) # height, width

            # Concatenate along the Channel (Batch , Channel, Height, Width )
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip) # DoubleConv

        return self.final_conv(x) # 1x1 Conv


def test(): # Down - BottleNeck - Up - Final Conv --- Size Check 
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels= 1, out_channels= 1)
    preds = model(x)

    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
