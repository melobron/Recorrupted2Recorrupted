import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class ImageNetGray(Dataset):
    def __init__(self, std=25, alpha=0.5, data_dir='../all_datasets/ImageNet_1000_Gray/', train=True, transform=None):
        super(ImageNetGray, self).__init__()

        self.std = std
        self.alpha = alpha

        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')

        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.
        noisy = clean + np.random.randn(*clean.shape) * (self.std/255.)

        input_img = noisy + np.random.randn(*clean.shape) * (self.std/255.) * self.alpha
        target_img = noisy - np.random.randn(*clean.shape) * (self.std/255.) / self.alpha

        clean, noisy = self.transform(clean), self.transform(noisy)
        input_img, target_img = self.transform(input_img), self.transform(target_img)
        clean, noisy = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor)
        input_img, target_img = input_img.type(torch.FloatTensor), target_img.type(torch.FloatTensor)
        return {'clean': clean, 'noisy': noisy, 'input': input_img, 'target': target_img}

    def __len__(self):
        return len(self.clean_paths)







