
from config import batch_size
from torch.utils.data import DataLoader
from crop_dataset import CropDataset
from torchvision import transforms


def data_loader(train):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])
    crop_data = CropDataset(train=train,transform=transformer)


    loader = DataLoader(
        crop_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader

