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

from torch.autograd import Variable

import scipy.io as scio
import torchvision.transforms as transforms

import argparse

from lib import dataload
from lib.AF import AF
from lib.MNet import MNet
from lib.Hydraplus import HP

import pdb
import pickle
from att_vis import att_plot

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', help="choose model", choices=['AF1', 'AF2', 'AF3', 'HP', 'MNet'], required=True)
    parser.add_argument('-p', help='wight file path', required=True)

    parser.add_argument('-att',
                        help='attention results',
                        choices=['no_att',      # no attention test
                                 'img_save',   # save attention results as img
                                 'img_show',   # plot attention
                                 'pkl_save'],  # save attention results as pickle file
                        default='no_att')
    args = parser.parse_args()

    return args


def model_pred(data_input, model_name, img_name, network, att_mode):
    if att_mode == 'no_att':
        outputs = network(data_input)
    else:
        if model_name == 'HP':
            att1, att2, att3, outputs = network(data_input)
            out_dict = {
                "filename": img_name[0],
                'AF1': att1[0].detach().numpy(),
                'AF2': att2[0].detach().numpy(),
                'AF3': att3[0].detach().numpy()
            }

        elif 'AF' in model_name:
            outputs, att = network(data_input)
            out_dict = {
                "filename": img_name[0],
                model_name: att[0].detach().numpy()
            }

        if att_mode == 'pkl_save':
            pickle.dump(
                out_dict,
                open('result/att_output_' + model_name + '.pkl', 'ab')  # append
            )
        else:
            att_plot(model_name, out_dict, att_mode)

    return outputs


def main():
    args = parse_args()

    mytransform = transforms.Compose([

        transforms.Resize((299, 299)),
        transforms.ToTensor()
        ]
    )

    # torch.utils.data.DataLoader
    test_set = dataload.myImageFloder(root="./data/PA-100K/release_data/release_data",
                                      label="./data/PA-100K/annotation/annotation.mat",
                                      transform=mytransform,
                                      mode='test')
    imgLoader = torch.utils.data.DataLoader(
             test_set,
             batch_size=1, shuffle=True, num_workers=2)

    print('image number in test set: {}'.format(len(test_set)))

    mat = scio.loadmat("./data/PA-100K/annotation/annotation.mat")
    att = mat["attributes"]

    count = 0
    classes = []
    for c in att:
        classes.append(c[0][0])
        count = count + 1

    path = args.p

    if args.att == 'no_att':
        if 'AF' in args.m:
            net = AF(af_name=args.m)
        elif args.m == 'HP':
            net = HP()
        elif args.m == 'MNet':
            net = MNet()
    else:
        if 'AF' in args.m:
            net = AF(att_out=True, af_name=args.m)
        elif args.m == 'HP':
            net = HP(att_out=True)

    net.load_state_dict(torch.load(path))
    print("para_load_done")

    net.eval()
    net.cuda()
    dataiter = iter(imgLoader)

    count = 0

    TP = [0.0] * 26
    P  = [0.0] * 26
    TN = [0.0] * 26
    N  = [0.0] * 26

    Acc = 0.0
    Prec = 0.0
    Rec = 0.0

    if args.att == 'pkl_save':
        # overwrite the previous pickle file
        pkl_file = 'result/att_output_' + args.m + '.pkl'
        if os.path.exists(pkl_file):
            os.remove(pkl_file)

    while count < 10:  # todo: full test set: len(test_set)
        images, labels, filename = dataiter.next()
        inputs, labels = Variable(images, volatile=True).cuda(), Variable(labels).cuda()

        outputs = model_pred(data_input=inputs,
                             model_name=args.m,
                             img_name=filename,
                             network=net,
                             att_mode=args.att)

        Yandf = 0.1
        Yorf = 0.1
        Y = 0.1
        f = 0.1

        i = 0

        for item in outputs[0]:
            if item.data.item() > 0:
                f = f + 1
                Yorf = Yorf + 1
                if labels[0][i].data.item() == 1:
                    TP[i] = TP[i] + 1
                    P[i] = P[i] + 1
                    Y = Y + 1
                    Yandf = Yandf + 1
                else :
                    N[i] = N[i] + 1
            else :
                if labels[0][i].data.item() == 0:
                    TN[i] = TN[i] + 1
                    N[i] = N[i] + 1
                else:
                    P[i] = P[i] + 1
                    Yorf = Yorf + 1
                    Y = Y + 1
            i = i + 1
        Acc = Acc +Yandf/Yorf
        Prec = Prec + Yandf/f
        Rec = Rec + Yandf/Y
        if count % 1 == 0:
            print('test on {}th img'.format(count))
        count = count + 1

    Accuracy = 0
    print(TP)
    print(TN)
    print(P)
    print(N)
    for l in range(26):
        print("%s : %f" %(classes[l], (TP[l]/P[l] + TN[l]/N[l])/2))
        Accuracy = TP[l]/P[l] + TN[l]/N[l] + Accuracy
    meanAccuracy = Accuracy / 52

    print("path: %s mA: %f" % (path, meanAccuracy))

    Acc = Acc/10000
    Prec = Prec/10000
    Rec = Rec/10000
    F1 = 2 * Prec * Rec / (Prec + Rec)

    print("ACC: %f" % Acc)
    print("Prec: %f" % Prec)
    print("Rec: %f" % Rec)
    print("F1: %f" % F1)


if __name__ == '__main__':
    main()
