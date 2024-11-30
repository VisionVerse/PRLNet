# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/5/16 14:52

"""
import argparse

# RGB-T
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
parser.add_argument("--trainsize", type=int, default=384, help="training dataset size")
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")

parser.add_argument("--epoch", type=int, default=300, help="epoch number")  # 300
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")  # default, 5e-5
parser.add_argument(
    "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
)
parser.add_argument(
    "--decay_epoch", type=int, default=100, help="every n epochs decay learning rate"
)

# mod = "large"
mod = "base"
parser.add_argument(
    "--load",
    type=str,
    default=f"/home/shared/swinT_ckpt/swin_{mod}_patch4_window12_384_22k.pth",
    help="train from checkpoints",
)
parser.add_argument(
    "--load_pre",
    type=str,
    default="",
    help="train from checkpoints",
)
parser.add_argument("--gpu_id", type=str, default="0", help="train use gpu")
parser.add_argument("--parallel", default=True)


root = '/home/hzhou/data/RGBT_SOD/'
parser.add_argument(
    "--rgb_root",
    type=str,
    default=f"{root}/VT5000_clear/Train/RGB/",
    help="the training rgb images root",
)
parser.add_argument(
    "--depth_root",
    type=str,
    default=f"{root}/VT5000_clear/Train/T/",
    help="the training thermal images root",
)
parser.add_argument(
    "--gt_root",
    type=str,
    default=f"{root}/VT5000_clear/Train/GT/",
    help="the training gt images root",
)
parser.add_argument(
    "--edge_root",
    type=str,
    default=f"{root}/VT5000_clear/Train/GT/",
    help="the training edge images root",
)

parser.add_argument(
    "--test_rgb_root",
    type=str,
    default=f"{root}/VT5000_clear/Test/RGB/",
    help="the test gt images root",
)
parser.add_argument(
    "--test_depth_root",
    type=str,
    default=f"{root}/VT5000_clear/Test/T/",
    help="the test gt images root",
)
parser.add_argument(
    "--test_gt_root",
    type=str,
    default=f"{root}/VT5000_clear/Test/GT/",
    help="the test gt images root",
)
parser.add_argument(
    "--test_edge_root",
    type=str,
    default=f"{root}/VT5000_clear/Test/Edge/",
    help="the test edge images root",
)


parser.add_argument(
    "--save_path",
    type=str,
    default="./PRLNet/",  # lr=5e-5

    help="the path to save models and logs",
)
opt = parser.parse_args()
