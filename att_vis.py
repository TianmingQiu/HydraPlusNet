import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import os


def pkl2list(pkl_file):
    att_outputs_list = []
    with open(pkl_file, 'rb') as file:
        while True:
            try:
                att_dict = pickle.load(file)
            except:
                break
            att_outputs_list.append(att_dict)

    return att_outputs_list


def att_plot(model_nm, att_dict, plot_mode):
    input_img = cv2.imread("./data/PA-100K/release_data/release_data/" + att_dict["filename"])
    att_size = (input_img.shape[1], input_img.shape[0])
    att_level_num = len(att_dict) - 1
    att_channel_num = 8
    # plot initial image
    plt.subplot(att_level_num + 1, att_channel_num, 1)
    plt.imshow(input_img)
    plt.axis('off')
    # create a color map
    color_map = np.uint8([[250], [180], [120], [60], [0]])
    plt.subplot(att_level_num + 1, att_channel_num, 2)
    plt.imshow(cv2.resize(color_map, att_size))
    plt.axis('off')

    # plot attention
    for att_idx in range(att_level_num):
        for channel_idx in range(att_channel_num):
            if model_nm == 'HP':
                att_pm = att_dict['AF'+str(att_idx+1)]
                att = np.uint8(255 * cv2.resize(att_pm[channel_idx], att_size) / np.max(att_pm))
                # heat_map = cv2.applyColorMap(att, cv2.COLORMAP_SUMMER)
            else:
                att = np.uint8(255 * cv2.resize(att_dict[model_nm][channel_idx], att_size) / np.max(att_dict[model_nm]))

            plt.subplot(att_level_num + 1, att_channel_num, (att_idx + 1) * 8 + channel_idx + 1)
            plt.imshow(att)
            plt.axis('off')

    if plot_mode == 'img_show':
        plt.axis('off')
        plt.show()
    elif plot_mode == 'img_save':
        if not os.path.exists("result/att_img_" + model_nm):
            os.mkdir("result/att_img_" + model_nm)
        plt.axis('off')
        plt.savefig("result/att_img_" + model_nm + '/' + att_dict["filename"][:-4] + '.png')  # To save figure


if __name__ == "__main__":
    torch.cuda.set_device(3)
    model_name = 'HP'
    output = pkl2list('result/att_output_' + model_name + '.pkl')
    for att_dict in output:
        att_plot(model_name, att_dict, plot_mode='img_save')


