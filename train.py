# -*- coding: utf-8 -*-

import os
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.device_count()

from datetime import datetime
import sys
from tqdm import tqdm

sys.path.append("./models")

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from models.PRLNetV4 import PRLNet

from dataV2 import SalObjDataset, test_dataset
from utils import clip_gradient, adjust_lr
from options import opt
from tools import ramps
from tools.mag_angle_loss import EuclideanAngleLossWithOHEM


cudnn.benchmark = True

print(opt.save_path.split("/")[-1])
if not os.path.isdir(opt.save_path):
    os.mkdir(opt.save_path)


# ======== build the model
model = PRLNet()

if opt.load is not None:
    model.load_pre(opt.load)
    print("load model from ", opt.load)

if opt.parallel and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model.cuda()


# ======== load data ========
print("load data...")
train_dataset = SalObjDataset(
    opt.rgb_root, opt.gt_root, opt.depth_root, opt.edge_root, trainsize=opt.trainsize
)
train_loader = DataLoader(
    train_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
test_loader = test_dataset(
    opt.test_rgb_root, opt.test_gt_root, opt.test_depth_root, opt.trainsize
)
total_step = len(train_loader)



step = 0
writer = SummaryWriter(opt.save_path + "summary")

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
BCE = torch.nn.BCELoss()
MSE = torch.nn.MSELoss()
DF = EuclideanAngleLossWithOHEM()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return 1 * ramps.sigmoid_rampup(epoch, 40)


params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
oper = "Adam"

# train function
def train(train_loader, model, optimizer, epoch):
    global step

    model.train()

    sal_loss_all = 0
    loss_all = 0
    epoch_step = 0

    tqdm_ = enumerate(train_loader, start=1)
    for i, data in tqdm(tqdm_, total=len(train_loader)):

        # if i > 2:
        #     break

        (images, thermal, gts, edge, gt_df, dist_map) = data

        images = images.cuda()  # torch.Size([8, 3, 384, 384])
        gts = gts.cuda()  # torch.Size([8, 1, 384, 384])
        thermal = thermal.repeat(1, 3, 1, 1).cuda()  # torch.Size([8, 3, 384, 384])
        edge = edge.cuda()  # torch.Size([8, 1, 384, 384])

        df_gt, dist_map = gt_df.cuda(), dist_map.unsqueeze(1).cuda()

        # ===========
        out, df, lst = model(images, thermal)  # V4

        # ============ 计算loss ============
        # (4) 显著性loss
        loss_sal = CE(out, gts)
        # loss_sal = get_saliency_smoothness(torch.sigmoid(out), gts, df)  # Ls

        loss_lst = MSE(lst, dist_map.float())
        loss_df = DF(df, df_gt, gts)

        loss = loss_sal + loss_df + loss_lst

        optimizer.zero_grad()
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1

        sal_loss_all += loss_sal.data
        loss_all += loss.data
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if i % 100 == 0 or i == total_step or i == 1:

            # Tensorboard
            writer.add_scalar("Loss/Loss_all", loss.data, global_step=step)
            writer.add_scalar("Loss/Sal", loss_sal.data, global_step=step)
            # writer.add_scalar("Loss/DF", loss_df.data, global_step=step)
            writer.add_scalar("Loss/LS", loss_lst.data, global_step=step)
            # writer.add_scalar("Loss/Cons", loss_cons.data, global_step=step)

            grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image("RGB", grid_image, step)
            grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image("Ground_truth", grid_image, step)
            res = out[0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image("res", torch.tensor(res), step, dataformats="HW")

    sal_loss_all /= epoch_step
    loss_all /= epoch_step

    writer.add_scalar("Loss-epoch/all", loss_all, global_step=epoch)
    writer.add_scalar("Loss-epoch/sal", sal_loss_all, global_step=epoch)

    if epoch % 5 == 0:
        torch.save(model.state_dict(),  opt.save_path + '/PRLNet_epoch_{}.pth'.format(epoch))


if __name__ == "__main__":

    start_datetime = datetime.now().replace(microsecond=0)

    print("Start train...")
    print(f"===== {opt.save_path} =====")

    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(0, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar("learning_rate", cur_lr, global_step=epoch)

        train(train_loader, model, optimizer, epoch)
