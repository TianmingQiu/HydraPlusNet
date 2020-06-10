"""
Copyright: Tianming Qiu
28th May 2020
"""

import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torch.nn as nn

import scipy.io as scio
import torchvision.transforms as transforms

from lib.AF import AF
from lib.MNet import MNet
from lib.Hydraplus import HP
from lib import dataload

from torch.autograd import Variable
import argparse
import logging
import pdb
import numpy as np
# from visdom import Visdom
#
# viz = Visdom()
# win = viz.line(
#     Y=np.array([0.2]),
#     name="1"
# )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help="choose model", choices=['MNet', 'AF1', 'AF2', 'AF3', 'HP'])

    # pre-trained checkpoint
    parser.add_argument('-r', dest='r', help="resume training", default=False)
    parser.add_argument('-checkpoint', dest='checkpoint', help="load weight path", default=None)
    parser.add_argument('-mpath', dest='mpath', help="load MNet weight path", default=None)
    parser.add_argument('-af1path', dest='af1path', help="load AF1 weight path", default=None)
    parser.add_argument('-af2path', dest='af2path', help="load AF2 weight path", default=None)
    parser.add_argument('-af3path', dest='af3path', help="load AF3 weight path", default=None)

    # training hyper-parameters
    parser.add_argument('-nw', dest='nw', help="number of workers for dataloader",
                        default=0, type=int)
    parser.add_argument('-bs', dest='bs', help="batch size",
                        default=1, type=int)
    parser.add_argument('-lr', dest='lr', help="learning rate",
                        default=0.001, type=float)
    parser.add_argument('-mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')

    args = parser.parse_args()

    return args


def checkpoint_save(args_m, state_dict, epoch):
    # if not os.path.exists("checkpoint/" + args_m):
    #     os.mkdir("checkpoint/" + args_m)
    save_path = "./checkpoint/" + args_m + "_epoch_{}".format(epoch)
    torch.save(state_dict, save_path)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


def main():
    args = parse_args()

    mytransform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.Resize((299, 299)),  # TODO: resize for different input image size
         transforms.ToTensor()]
    )

    # torch.utils.data.DataLoader
    data_set = dataload.myImageFloder(
        root="./data/PA-100K/release_data/release_data",
        label="./data/PA-100K/annotation/annotation.mat",
        transform=mytransform,
        mode='train'
    )

    imgLoader = torch.utils.data.DataLoader(
        data_set,
        batch_size=args.bs, shuffle=True, num_workers=args.nw
    )

    print('image numbers {}'.format(len(data_set)))

    # define the training model
    if args.m == 'MNet':
        net = MNet()
        if not args.r:
            net.apply(weight_init)
    elif 'AF' in args.m:
        net = AF(af_name=args.m)
        if not args.r:
            net.MNet.load_state_dict(torch.load(args.mpath))

        for param in net.MNet.parameters():
            param.requires_grad = False
    elif args.m == 'HP':
        net = HP()
        if not args.r:
            net.MNet.load_state_dict(torch.load(args.mpath))
            net.AF1.load_state_dict(torch.load(args.af1path))
            net.AF2.load_state_dict(torch.load(args.af2path))
            net.AF3.load_state_dict(torch.load(args.af3path))

        for param in net.MNet.parameters():
            param.requires_grad = False

        for param in net.AF1.parameters():
            param.requires_grad = False

        for param in net.AF2.parameters():
            param.requires_grad = False

        for param in net.AF3.parameters():
            param.requires_grad = False

    # resume training and load the checkpoint from last training
    start_epoch = 0
    if args.r:
        net.load_state_dict(torch.load(args.checkpoint))

        # reset the start epoch
        numeric_filter = filter(str.isdigit, args.checkpoint)
        numeric_string = "".join(numeric_filter)
        start_epoch = int(numeric_string) + 1

    net.cuda()
    if args.mGPUs:
        net = nn.DataParallel(net)

    net.train()

    # weights of different classes
    loss_cls_weight = [1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588,
                       # [u'Female', u'AgeOver60', u'Age18-60', u'AgeLess18',
                       1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586,
                       # u'Front', u'Side', u'Back', u'Hat',
                       2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033,
                       # u'Glasses', u'HandBag', u'ShoulderBag', u'Backpack'
                       2.692210221689372, 1.5128949487853383, 1.7967419553099035, 2.5832221110933764,
                       # u'HoldObjectsInFront', u'ShortSleeve', u'LongSleeve', u'UpperStride',
                       2.3302195718894034, 2.438480257574324, 2.6012705532709526, 2.704589108443237,
                       # u'UpperLogo', u'UpperPlaid', u'UpperSplice', u'LowerStripe',
                       2.6704246374231753, 2.6426970354162505, 1.3377813061118478, 2.284449325734624,
                       # u'LowerPattern', u'LongCoat', u'Trousers', u'Shorts',
                       2.417810793601295, 2.7015143874115033]  # u'Skirt&Dress', u'boots']

    weight = torch.Tensor(loss_cls_weight)

    criterion = nn.BCEWithLogitsLoss(weight=weight)  # TODO:1.learn 2. weight
    criterion.cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                momentum=0.9)

    running_loss = 0.0
    logging.basicConfig(level=logging.DEBUG,
                        filename='./result/training_log/' + args.m + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    for epoch in range(start_epoch, 1000):
        for i, data in enumerate(imgLoader, 0):
            # get the inputs
            inputs, labels, _ = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            # running_loss += loss.data[0]
            # running_loss += loss.item()

            if i % 200 == 0:  # todo: print every 1000 mini-batches
                logger.info('[  %d  %5d] loss: %.6f' % (epoch, i + 1, loss))
                # viz.updateTrace(
                #     X=np.array([epoch+i/5000.0]),
                #     Y=np.array([running_loss]),
                #     win=win,
                #     name="1"
                # )
                # (epoch + 1, i + 1, running_loss / 2000))
                # running_loss = 0.0

        if epoch % 5 == 0:
            if args.mGPUs:
                checkpoint_save(args.m, net.module.state_dict(), epoch)
            else:
                checkpoint_save(args.m, net.state_dict(), epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.95


if __name__ == '__main__':
    main()
