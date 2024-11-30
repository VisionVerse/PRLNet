import os
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance

import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.SDM import my_compute_sdf
from tools.direction_field import direct_field


def cv_random_flip(img, label, depth, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, depth, edge


def randomCrop(image, label, depth, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return (
        image.crop(random_region),
        label.crop(random_region),
        depth.crop(random_region),
        edge.crop(random_region),
    )


def randomRotation(image, label, depth, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, depth, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test.
# If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(Dataset):
    def __init__(
        self,
        image_root,
        gt_root,
        thermal_root,
        edge_root,
        trainsize,
        df_norm=True,
        boundary=True,
        df_used=True,
    ):
        self.df_used = df_used
        self.df_norm = df_norm
        self.boundary = boundary

        self.trainsize = trainsize
        self.images = [
            image_root + f for f in os.listdir(image_root) if f.endswith(".jpg")
        ]

        self.gts = [
            gt_root + f
            for f in os.listdir(gt_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        self.thermal = [
            thermal_root + f
            for f in os.listdir(thermal_root)
            if f.endswith(".bmp") or f.endswith(".png") or f.endswith(".jpg")
        ]
        self.edges = [
            edge_root + f
            for f in os.listdir(edge_root)
            if f.endswith(".bmp") or f.endswith(".png") or f.endswith(".jpg")
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.thermal = sorted(self.thermal)
        self.edges = sorted(self.edges)
        self.size = len(self.images)
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)),
             transforms.ToTensor(),  # 有归一化功能
             ]
        )
        self.thermal_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )
        self.edges_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # print(self.images[index])
        image = rgb_loader(self.images[index])
        gt = binary_loader(self.gts[index])  # <class 'tuple'>: (617, 460)
        # if np.array(gt).min() > 0:
        #     print(np.array(gt).min())
        thermal = binary_loader(self.thermal[index])
        edge = binary_loader(self.edges[index])

        image, gt, thermal, edge = cv_random_flip(image, gt, thermal, edge)
        image, gt, thermal, edge = randomCrop(image, gt, thermal, edge)
        image, gt, thermal, edge = randomRotation(image, gt, thermal, edge)
        image = colorEnhance(image)
        # gt = randomGaussian(gt)
        # gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)  # torch.Size([1, 384, 384]), 有归一化功能
        # print(gt.numpy().min())
        thermal = self.thermal_transform(thermal)
        edge = self.edges_transform(edge)

        if self.df_used:
            gt_df = direct_field(gt.numpy()[0], norm=self.df_norm)
            gt_df = torch.from_numpy(gt_df)
        else:
            gt_df = None

        if self.boundary:
            dist_map = my_compute_sdf(gt.numpy()[0])
        else:
            dist_map = None

        return image, thermal, gt, edge, gt_df, dist_map
        # return image, thermal, gt, edge, gt_df, dist_map, self.images[index]

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        edges = []
        for img_path, gt_path, depth_path, edge_path in zip(
            self.images, self.gts, self.depths, self.edges
        ):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and gt.size == depth.size and edge.size == img.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.depths = depths
        self.edges = edges

    def resize(self, img, gt, depth, edge):
        assert img.size == gt.size and gt.size == depth.size and edge.size == img.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (
                img.resize((w, h), Image.BILINEAR),
                gt.resize((w, h), Image.NEAREST),
                depth.resize((w, h), Image.NEAREST),
                edge.resize((w, h), Image.NEAREST),
            )
        else:
            return img, gt, depth, edge

    def __len__(self):
        return self.size


def rgb_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def binary_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("L")
        img = np.array(img)
        if img.min() > 0 or img.max() < 255:
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255

        img = Image.fromarray(np.uint8(img))
        return img


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f for f in os.listdir(image_root) if f.endswith(".jpg")
        ]
        self.gts = [
            gt_root + f
            for f in os.listdir(gt_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.depths = [
            depth_root + f
            for f in os.listdir(depth_root)
            if f.endswith(".bmp") or f.endswith(".png") or f.endswith(".jpg")
        ]
        # self.edges = [edge_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)),
             transforms.ToTensor()]
        )
        # self.edges_transform = transforms.Compose(
        #     [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = binary_loader(self.gts[self.index])

        depth = binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)

        name = self.images[self.index].split("/")[-1]

        # image_for_post = rgb_loader(self.images[self.index])
        # image_for_post = image_for_post.resize(gt.size)

        if name.endswith(".jpg"):
            name = name.split(".jpg")[0] + ".png"

        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name,

    def __len__(self):
        return self.size

