import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import random
import numpy as np
import glob
from PIL import Image
import pickle as pkl

Data_frame = './data/frames'
Data_info = './data/info'
C, H, W = 3, 224, 224   # input data dims;
input_resize = 455, 256   # image resizing dims;
num_frames = 103   # num of frames in a single video
segment_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]   # beginning frames of the 10 segments

def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()
        self.mode = mode  # train or test
        self.args = args
        self.annotations = pkl.load(open(os.path.join(Data_info, 'augmented_final_annotations_dict.pkl'), 'rb'))
        self.keys = pkl.load(open(os.path.join(Data_info, f'{self.mode}_split_0.pkl'), 'rb'))

    def get_imgs(self, key):
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_list = sorted((glob.glob(os.path.join(Data_frame,
                                                    str('{:02d}_{:02d}'.format(key[0], key[1])),
                                                    '*.jpg'))))
        sample_range = np.arange(0, num_frames)
        # temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = random.randint(0, self.args.temporal_aug)
            sample_range += temporal_aug_shift
        # spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0, 1)
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            if self.mode == 'train':
                images[j] = load_image_train(image_list[i], hori_flip, transform)
            if self.mode == 'test':
                images[j] = load_image(image_list[i], transform)
        return images

    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}
        data['video'] = self.get_imgs(key)
        data['final_score'] = self.annotations.get(key).get('final_score')
        data['difficulty'] = self.annotations.get(key).get('difficulty')
        data['judge_scores'] = self.annotations.get(key).get('judge_scores')
        return data

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders