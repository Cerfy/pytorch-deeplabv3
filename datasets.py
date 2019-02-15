import os
import os.path as osp
import numpy as np
import torch
import torchvision
import cv2
from torch.utils import data


class BerkeleyDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mean=(128,128,128), 
                    scale=True, mirror=True, ignore_label =255, train=True):
        self.root = root
        self.list_path = list_path
        self.max_iters = max_iters
        self.scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label
        self.mean = mean
        self.train = train

        # Get image file names
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []


        if self.train == True:
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/100k/train/{}.jpg".format(name))
                label_file = osp.join(self.root, "drivable_maps/color_labels/train/{}_drivable_color.png".format(name))
                self.files.append({
                    "img": img_file, 
                    "label": label_file, 
                    "name": name})
        # For Berkeley validation dataset
        else:
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/100k/val/{}.jpg".format(name))
                label_file = osp.join(self.root, "drivable_maps/color_labels/val/{}_drivable_color.png".format(name))
                self.files.append({
                    "img": img_file, 
                    "label": label_file, 
                    "name": name})
        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        size = image.shape

        # Height = axis 0 (y) , Width = axis 1 (x)
        image = cv2.resize(image, None, fx=321/size[1], fy=321/size[0], interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=321/size[1], fy=321/size[0], interpolation = cv2.INTER_NEAREST)
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image -= self.mean

        # cv2.imshow("image", image)
        # cv2.waitKey(5000)


        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        '''
        Berkeley Dataset has 4 different labels, 
        they have differnt shades of red and blue but they mean the same thing. 
        The following are the label values after converting to grayscale.
        Red = 29, Blue = 76, Light Red = 123, Light Blue = 165
        '''
        
        label[label == 29] = 1 # Red labels
        label[label == 123] = 1 # Light Red Labels
        label[label == 76] = 2 # Blue labels
        label[label == 165] = 2 # Light Blue Labels

        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), name, np.array(size)



# if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import sys
    # sys.path.append('../dataset')
    # dst = BerkeleyDataset("./BDD_Deepdrive/bdd100k/", './dataset/list/BDD_train.txt')

    # # Show Augmented BerkeleyDataSet
    # trainloader = data.DataLoader(dst, batch_size=4)
    # for i, data in enumerate(trainloader):
    #     imgs, labels,_,_ = data
    #     if i == 0:
    #         img = torchvision.utils.make_grid(imgs).numpy()
    #         img = np.transpose(img, (1, 2, 0))
    #         img = img[:, :, ::-1]
    #         plt.imshow(img)
    #         plt.show()






