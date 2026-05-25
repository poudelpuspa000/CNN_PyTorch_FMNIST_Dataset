import torch
import torchvision
import torchvision.transforms as transforms

# -----------------------------------
# Transformations
# -----------------------------------

normalTransformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

augmentationTransformation = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------------
# Load Data
# -----------------------------------

def get_dataloaders(useAugmentation=False, batchSize=128):

    transformation = (
        augmentationTransformation
        if useAugmentation
        else normalTransformation
    )

    trainDataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transformation
    )

    trainLoader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=batchSize,
        shuffle=True
    )

    testDataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=normalTransformation
    )

    testLoader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=batchSize,
        shuffle=False
    )

    return trainLoader, testLoader


# Classes


classes = (
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
)
