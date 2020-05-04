import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

def pkl2list(pkl_file):
    att_outputs_list = []
    with open(pkl_file, 'rb') as file:
        while True:
            try:
                att_dict = pickle.load(file)
            except:
                break
            att_dict["filename"] = att_dict["filename"][0]
            att_dict["alpha1"] = att_dict["alpha1"][0].cpu().detach().numpy()
            att_dict["alpha2"] = att_dict["alpha2"][0].cpu().detach().numpy()
            att_dict["alpha3"] = att_dict["alpha3"][0].cpu().detach().numpy()
            att_outputs_list.append(att_dict)

    return att_outputs_list

def vis_att(att_dict):
    input_img = cv2.imread("data/PA-100K/release_data/release_data/" + att_dict["filename"])
    att_size = (input_img.shape[1], input_img.shape[0])
    plt.subplot(4, 8, 1), plt.imshow(input_img)
    for att_idx in range(3):
        for channel_idx in range(8):
            att1 = cv2.resize(att_dict["alpha" + str(att_idx+1)][channel_idx], att_size) / np.max(att_dict["alpha1"][channel_idx])
            heat_map = cv2.applyColorMap(np.uint8(255 * att1), cv2.COLORMAP_JET)

            plt.subplot(4, 8, 8 * (att_idx + 1) + channel_idx + 1), plt.imshow(heat_map)
    plt.savefig('att_result/' + att_dict["filename"] + '.png')  # To save figure
    # plt.show()  # To show figure


if __name__ == "__main__":
   output = pkl2list('attention_output.pkl')
   for att_dict in output:
       vis_att(att_dict)


