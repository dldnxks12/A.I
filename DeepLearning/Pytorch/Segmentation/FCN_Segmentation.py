from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import wget

# fcn model load 
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# Image Download 
wget.download("https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19/pinyon-jay-bird.jpg")

# Image Visualize

img = Image.open("./pinyon-jay-bird.jpg")
plt.imshow(img)
plt.show()

# data transform
import torchvision.transforms as T

transform = T.Compose( [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
transformed_img = transform(img).unsqueeze(0)

# See the model output
out = fcn(transformed_img)['out']
print(out.shape)

output = torch.argmax(out, dim = 0).detach().numpy()

print(output.shape)
print(np.unique(output))

# Make Color map --  2D image to RGB Image
def decode_segmap( image, nc = 21 ):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for i in range(nc):
        idx = image == i # 만일 해당 클래스랑 image 값이 같다면?
        
        r[idx] = label_colors[i,0] 
        g[idx] = label_colors[i,1]
        b[idx] = label_colors[i,2]
        
    rgb = np.stack([r,g,b], axis = 2)
    
    return rgb

rgb = decode_segmap(output)

plt.imshow(rgb)
plt.show()
