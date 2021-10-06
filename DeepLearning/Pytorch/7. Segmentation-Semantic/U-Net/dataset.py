import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transform
import numpy as np

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform.Compose([transform.ToTensor(),
                                            transform.Resize((160, 240))])

        self.images = os.listdir(image_dir) # image_dir 경로에 있는 모든 file의 이름을 가져온다. (path X file name O)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])  # Rename as "path + file name" !
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif"))  # Rename as "path + file name" !
        '''
            _mask.gif 확장자 -> jpg로 바꾸기 , path + filename 결합하기 동시에 수행         
        '''
        # RGB Image로 열어서 numpy type으로 type casting
        image = np.array(Image.open(img_path).convert("RGB")) # 확실히 하기 위해 RGB Image로 열도록 명시
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # L : gray Scale

        image = self.transform(image)
        mask = self.transform(mask)

        # 0.0 ~ 255.0
        mask[mask == 255.0] = 1.0 # Sigmoid 를 마지막 Activation function으로 사용할 것이기 때문에 1의 값으로 맞추어주자.

        return image, mask


