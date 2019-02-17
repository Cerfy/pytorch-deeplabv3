import argparse

import torch
import torch.nn as nn
from torch.utils import data 
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from deeplabv3 import Res_Deeplab
from datasets import BerkeleyDataset
import random
import timeit
import matplotlib.pyplot as plt

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


### Define Constants ###
BATCH_SIZE = 2
DATA_DIRECTORY = 'D:/BDD_Deepdrive/bdd100k/'
DATA_LIST_PATH = './dataset/list/BDD_train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './BDD_Deepdrive_Snapshots/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()


# Calculate the loss of the entire network
def loss_calc(pred, label):
    label = label.long().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    return criterion(pred, label)

# Polynomial Learning Rate decay
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr



def main():
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    model = Res_Deeplab(num_classes=args.num_classes)
    # print(model) # Check number of classes

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
    #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
    # print i_parts
    # if not i_parts[1]=='layer5':
        if not i_parts[0]=='fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 

    model.load_state_dict(new_params)
    model.train()
    model.cuda()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        print("Creating Checkpoint Folder")
        os.makedirs(args.snapshot_dir)

    berkeleyDataset = BerkeleyDataset(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_loader = data.DataLoader(berkeleyDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) 

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate }], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)


    for i_iter, batch in enumerate(train_loader):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))


        loss = loss_calc(pred, labels)


        loss.backward()
        optimizer.step()

        print("Iteration = {} of {} completed, loss = {}".format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            print("Save Model ....")
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'BDD_' + str(i_iter) + '.pth'))
            break
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print("Taking snapshot ...")
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'BDD_' + str(i_iter) + '.pth'))
    end = timeit.default_timer()
    print("Total time it took was {} seconds".format(end-start))

if __name__ == '__main__':
    main()