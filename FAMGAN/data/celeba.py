from .base_dataset import BaseDataset
import os
import random
import numpy as np
import time
import math


class CelebADataset(BaseDataset):
    """docstring for CelebADataset"""
    def __init__(self):
        super(CelebADataset, self).__init__()
        
    def initialize(self, opt):
        super(CelebADataset, self).initialize(opt)

    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0   # norm to [0, 1]

    def make_dataset(self):
        # return all image full path in a list
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file
        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):
        #img_path = self.imgs_path[index]
        # load source image
        #src_img = self.get_img_by_path(img_path)
        #src_img_tensor = self.img2tensor(src_img)
        #src_aus = self.get_aus_by_path(img_path)

        img_path = 'datasets/celebA/imgs/000015.jpg'
        base='datasets/celebA/imgs/casme_sur_001.jpg'
        img_path_10003 = 'datasets/celebA/imgs/10003.jpg'
        src_img = self.get_img_by_path(img_path)
        src_img_tensor = self.img2tensor(src_img)
        src_aus = self.get_aus_by_path(img_path)
        
        # load target image
        # tar_img_path = random.choice(self.imgs_path)
        tar_img_path=self.imgs_path[index]

        tar_img = self.get_img_by_path(tar_img_path)
        tar_img_tensor = self.img2tensor(tar_img)
        tar_aus = self.get_aus_by_path(tar_img_path)
        base_aus= self.get_aus_by_path(base)
        # tar_aus[1]=tar_aus[1]*1.1
        # tar_aus[0]=tar_aus[0]*1.3
        #
        #tar_aus[4]=tar_aus[4]*1.2
        
        # tar_aus[8]*=1
        # tar_aus[9]*=1
        # tar_aus[11]*=1
        
        # print("tar",tar_img_path,"iniau\n",tar_aus)
        # tar_aus[3]=tar_aus[3]+0.17-abs(int(tar_img_path[-7:-4])-64)*0.01
        #
        # tar_aus=(tar_aus-base_aus)*0.8+src_aus
        #
        # tar_aus[3]=tar_aus[3]*0.30
        #
            
        #tar_aus[1]=tar_aus[1]*2+0.3
        #tar_aus[3]=tar_aus[3]*2
        # tar_aus[3]=tar_aus[3]
        # tar_aus[0]*=.5
        # tar_aus[1]*=.5
        # tar_aus[2]*=1
        
        #
        # tar_aus[6]=tar_aus[6]*1
        # tar_aus[7]=tar_aus[7]*1
        # tar_aus[9]=tar_aus[9]*1
        # tar_aus[4]=tar_aus[4]*1
        # tar_aus[5]=tar_aus[5]*1
        # tar_aus[8]=tar_aus[8]*1
        # tar_aus[10]=tar_aus[10]*1
        # tar_aus[11]=tar_aus[11]*1
        # tar_aus[13]=tar_aus[13]
        # tar_aus[14]=tar_aus[14]
        # tar_aus[15]=tar_aus[15]

        for idx,au in enumerate(tar_aus):
            if au<=0.03:
                tar_aus[idx]=0
        #print('source',src_aus)
        print("tar",tar_img_path,"tarau\n",tar_aus)
        if self.is_train and not self.opt.no_aus_noise:
            tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)

        # record paths for debug and test usage
        data_dict = {'src_img':src_img_tensor, 'src_aus':src_aus, 'tar_img':tar_img_tensor, 'tar_aus':tar_aus, \
                        'src_path':img_path, 'tar_path':tar_img_path}

        return data_dict
