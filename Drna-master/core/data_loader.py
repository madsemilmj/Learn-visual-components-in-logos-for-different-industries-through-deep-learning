import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import sys
from config import INPUT_SIZE

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, train = None, train_procent = 0.8, transform=None):
        self.transform = transform
        self.data_root = data_root
       # print(self.data_root)
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []
        self.intlabels = {"Food": 0, "Clothes": 1, "Institution": 2, "Accessories": 3, "Transportation": 4, "Electronic": 5, "Necessities": 6, "Cosmetic": 7, "Leisure": 8, "Medical": 9}
        for data in data_list:
            image_path = data[:-1]
            label = image_path.split('/')[0]
            self.img_paths.append(image_path)
            self.labels.append(label)
        

    def __getitem__(self, item):

     #   print(self.data_root)
        img_path, label= self.img_paths[item], self.labels[item]
        label = self.intlabels[label]
        #img_path_full = os.path.join(self.data_root, img_path)
        img_path_full = self.data_root+img_path
      #  print(img_path_full)
        img = Image.open(img_path_full).convert('RGB')
        # label = np.array(label,dtype='float32')
        #label = int(label)-100
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data

class TransformData(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)