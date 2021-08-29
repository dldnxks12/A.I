import torch
import matplotlib.pyplot as plt
from UNET import UNET
from PIL import Image
import torchvision.transforms as transform
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNET(in_channels= 3, out_channels= 1).to(device = DEVICE)

# load model
state = torch.load("./model.pth.tar")
model.load_state_dict(state["state_dict"])

# make sample test img
tf = transform.Compose([transform.ToTensor(), transform.Resize((160, 240))])

image = np.array(Image.open("./test.jpg").convert("RGB"))
image = tf(image)
image = image.unsqueeze(0).to(device = DEVICE)

# make prediction with test img
prediction = model(image)
prediction = prediction.squeeze(0)
prediction = prediction.squeeze(0)
prediction = torch.sigmoid(prediction)

# threshold
prediction[prediction > 0.5 ] = 1
prediction[prediction <= 0.5] = 0

im = Image.open("./test.jpg")
# visualize
prediction = prediction.cpu().detach().numpy()
plt.subplot(1,2,1)
plt.imshow(prediction)
plt.subplot(1,2,2)
plt.imshow(im)
plt.show()
