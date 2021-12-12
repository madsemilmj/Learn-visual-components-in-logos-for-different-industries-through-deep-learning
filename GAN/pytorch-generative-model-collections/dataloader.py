from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == "logo":
        os.chdir('/content')
        #zip_path = "/content/drive/MyDrive/DLSP/GAN/pytorch-generative-model-collections/data/TestData.zip"
        #!cp "{zip_path}" .
        #!unzip -q TestData.zip
        #!rm TestData.zip
        print(os.getcwd())
        print("holycow2")
        zip_path = "/content/drive/MyDrive/DLSP/GAN/pytorch-generative-model-collections/Data/TrainData.zip"
        os.system('cp "/content/drive/MyDrive/DLSP/GAN/pytorch-generative-model-collections/Data/TrainData.zip" .')
        os.system('unzip -q TrainData.zip')
        os.system('rm TrainData.zip')
        data = datasets.ImageFolder("TrainData", transform = transform)
        print(len(data))
        data_loader = DataLoader(
            data,
            batch_size=batch_size, shuffle=True)

    return data_loader