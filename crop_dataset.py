
from config import train_path,test_path,train_data,test_data
from config import width,height
from PIL import Image
import torch.utils.data as data


class CropDataset(data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.width = width
        self.height = height

        if self.train:
            self.train_data = train_data
            self.train_path = train_path
        else:
            self.test_data = test_data
            self.test_path = test_path

    def __getitem__(self, index):

        if self.train:
            img, target = Image.open(self.train_path + self.train_data[index]['image_id']),\
                          self.train_data[index]['disease_class']
        else:
            img, target = Image.open(self.test_path + self.test_data[index]['image_id']),\
                          self.test_data[index]['disease_class']

        img = img.resize((self.width, self.height), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


if __name__ == "__main__":

    crop_data = CropDataset(train=True)
    img,t = crop_data.__getitem__(0)
    print(img.shape,t)

