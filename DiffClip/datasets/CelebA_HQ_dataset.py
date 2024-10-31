from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import io

class MultiResolutionDataset(Dataset):
    # def __init__(self, path, transform, resolution=256):
    def __init__(self, path, transform,size=256, resolution=256):
        # self.images= os.listdir(path)
        # self.train_paths = [path +_ for _ in self.images]

        # self.resolution = resolution
        # self.transform = transform
        self.lmdb_path = path
        self.size = size  # 指定所需的图像尺寸，例如 256
        self.transform = transform

        # 加载 LMDB 中存储的图像总数量
        with lmdb.open(self.lmdb_path, readonly=True) as env:
            with env.begin() as txn:
                length = txn.get("length".encode("utf-8"))
                if length is None:
                    raise ValueError("Cannot find 'length' key in LMDB database.")
                self.length = int(length.decode("utf-8"))

    def __len__(self):
        # return len(self.train_paths)
         return self.length

    def __getitem__(self, index):
        
        # img_name = self.train_paths[index]

        # # buffer = BytesIO(img_bytes)
        # img = Image.open(img_name)
        # img=img.resize((256,256))
        # img = self.transform(img)

        # return img
        
        # 根据索引和指定的尺寸生成键
        key = f"{self.size}-{str(index).zfill(5)}".encode("utf-8")

        # 从 LMDB 中读取图像数据
        with lmdb.open(self.lmdb_path, readonly=True) as env:
            with env.begin() as txn:
                img_data = txn.get(key)
                if img_data is None:
                    raise ValueError(f"Image with key {key.decode('utf-8')} not found in LMDB database.")
                
                # 使用 Image.open 打开图像数据并转换为 RGB 格式
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # 应用变换（如果有）
        if self.transform is not None:
            img = self.transform(img)

        return img


################################################################################

def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDataset(os.path.join(data_root, 'train/'),
                                           train_transform, config.data.image_size)
    test_dataset = MultiResolutionDataset(os.path.join(data_root, 'test/'),
                                          test_transform, config.data.image_size)


    return train_dataset, test_dataset



