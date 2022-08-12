# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts import BtsModel
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *

from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

#python bts_test.py arguments_test_eigen.txt
#python3 bts_sequence.py --image_path depth1020 --dataset kitti --max_depth 80 --model_name bts_eigen_res2net101_v1b --checkpoint_path models/bts_eigen_res2net101_v1b/model-218500-best_abs_rel_0.05403 --encoder res2net101_v1b_bts

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--cfg',                       help='experiment configure file name',   required=True, type=str)
parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val

from segmentation.config import config, update_config
update_config(config, args)

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

class CityscapesMeta(object):
    def __init__(self):
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.ignore_label = 255

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap
def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    #print weight
    # weight = dict()
    # for name, para in model.named_parameters():
    #     # print('{}: {}'.format(name, para.shape))
    #     weight[name] = para

    # print("0",nn.Sigmoid()(weight['module.decoder.FFTskip0.0.dense.1.sigmas'])*93.0)
    # print("1",nn.Sigmoid()(weight['module.decoder.FFTskip1.0.dense.1.sigmas'])*93.0)
    # print("2",nn.Sigmoid()(weight['module.decoder.FFTskip2.0.dense.1.sigmas'])*93.0)
    # print("3",nn.Sigmoid()(weight['module.decoder.FFTskip3.0.dense.1.sigmas'])*93.0)
    # return

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []
    pred_seg = []

    start_time = time.time()
    img = []
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            # img.append((sample['image'].squeeze()*255.0).permute(1,2,0).numpy())
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, seg = model(image, focal)
            # print(seg.shape)
            # return
            # semantic_pred = get_semantic_segmentation(seg)
            # print(seg.cpu().numpy().squeeze().shape)
            # return
            pred_seg.append(seg.cpu().numpy().squeeze())
            # print(seg.cpu().numpy().squeeze().shape)
            # print(seg[0].cpu().numpy().squeeze().max(),seg[0].cpu().numpy().squeeze().min())
            # return
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            # print(lpg8x8[0].shape)  
            # return         
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.model_name
    
    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    meta_dataset = CityscapesMeta()
    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'kitti':
            date_drive = lines[s].split('/')[1]
            filename_pred_png = save_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
                -1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
        elif args.dataset == 'kitti_benchmark':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
        else:
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]
        
        rgb_path = os.path.join(args.data_path, lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path, lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)
        
        pred_s = pred_seg[s]
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        continue
        filename_image_png = filename_image_png.replace('.png', '.png')
        plt.imsave(filename_image_png,1.0-(pred_depth/np.linalg.norm(pred_depth)), cmap='plasma') #colorfull depth
        # continue

        # pred_s = pred_s[10:-1 - 9, 10:-1 - 9]
        # pred_s = (pred_s - pred_s.min()) / (pred_s.max() - pred_s.min())
        # # save_annotation(pred_s, save_name + '/rgb/', 'semantic_pred_%d' % s,
        # #                     add_colormap=True, colormap=meta_dataset.create_label_colormap())
        # filename_lpg_cmap_png = filename_cmap_png.replace('.png', 'last.png')   
        # plt.imsave(filename_lpg_cmap_png, pred_s, cmap='Greys')
        
        # continue
        for i in range(0,4):
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', 'up'+str(i)+'.png')
            plt.imsave(filename_lpg_cmap_png, pred_s[i][10:-1 - 9, 10:-1 - 9]/np.linalg.norm(pred_s[i][10:-1 - 9, 10:-1 - 9]), cmap='plasma')
        continue
       
        if args.save_lpg:
            # cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            else:        
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
    
    return


if __name__ == '__main__':
    test(args)
