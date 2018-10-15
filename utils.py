
from config import batch_size
from torch.utils.data import DataLoader
from crop_dataset import CropDataset
from torchvision import transforms as tfs


def data_loader(train):

    train_tfs = tfs.Compose([
            # tfs.Resize(120),  #随机比例缩放
            # tfs.RandomHorizontalFlip(),  #随机的水平和竖直方向翻转
            # tfs.RandomCrop(96),  #随机位置截取
            tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),  #亮度、对比度和颜色的变化
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    test_tfs= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    crop_data = CropDataset(train=train,transform=train_tfs if train else test_tfs)

    loader = DataLoader(
        crop_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return loader

