from random import randint
from torchvision import transforms

from timm.models.layers import to_2tuple

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from imageCS_utils.YCbCr import rgb2ycbcr

class LabDataset(Dataset):
    def __init__(self, dataset_root, dtype, crop_size=0, channels=1, dataset_len=-1) -> None:
        assert dtype in ["train", "val", "test"]
        super().__init__()
        self.dataset_len = dataset_len
        self.crop_size = to_2tuple(crop_size)
        self.dataset_root = dataset_root
        self.channels = channels
        self.dtype = dtype

        self._init_dataset()

    def _init_dataset(self):
        trans_list = []

        # >>> trans list
        if self.dtype == "train":
            trans_list.append(transforms.RandomCrop(size=self.crop_size, pad_if_needed=True, padding_mode="reflect"))
            trans_list.append(transforms.RandomHorizontalFlip())
        trans_list.append(transforms.ToTensor())
        # <<< trans list
        
        trans = transforms.Compose(trans_list)
        self.datasets = ImageFolder(root=self.dataset_root, transform=trans)
        self.real_len = len(self.datasets)

        self.random_index = False
        if self.dataset_len <= 0:
            self.dataset_len = self.real_len
        elif self.dataset_len < self.real_len:
            self.random_index = True

    def index_map(self, index):
        if self.random_index:
            index = randint(0, self.real_len - 1)
        else:
            index = index % self.real_len
        return index
    
    def get_data(self, index):
        (x, _) = self.datasets[index]

        if self.channels == 1:
            x = rgb2ycbcr(x, range=1)
            x = x[0:1, :, :]

        return x
    
    def __getitem__(self, index):
        index = self.index_map(index)
        x = self.get_data(index)
        return x
    
    def __len__(self):
        return self.dataset_len
