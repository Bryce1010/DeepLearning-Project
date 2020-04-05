from __future__ import division
import sys
import os

import warnings

from fpn_resnet1 import CSRNet
from utils import save_checkpoint
from rate_model import RATEnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import argparse
import json
import cv2
import dataset
import time
import math
from PIL import Image
from image import *
from centerloss import CenterLoss
import scipy.io as io
import scipy
from scipy.ndimage.filters import gaussian_filter
from find_couter import findmaxcontours
from functools import partial
import pickle



parser = argparse.ArgumentParser(description='PyTorch CSRNet')
#
# parser.add_argument('train_json', metavar='TRAIN',
#                     help='path to train json')
# parser.add_argument('test_json', metavar='TEST',
#                     help='path to test json')
#
# parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
#                     help='path to the pretrained model')
#
# parser.add_argument('gpu',metavar='GPU', type=str,
#                     help='GPU id to use.')
#
# parser.add_argument('task',metavar='TASK', type=str,
#                     help='task id to use.')
# parser.add_argument('density_value',metavar='DENSITY_VALUE',type=float,help='density value threthod')
warnings.filterwarnings('ignore')


def main():
    global args, best_prec1

    best_prec1 = 1e6
    orginal_best_prec1 = 1e6
    lr_cent = 1e-3
    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr =   0.5*1e-5
    args.rate_lr = 1e-6
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 800
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 100
    args.threshold = 1.04
    args.maxthreshold = 1.9
    args.density_value = 3

    args.task = "refineattention_demo"

    with open('./train.npy', 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open('./val.npy', 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    with open('./test.npy', 'rb') as outfile:
        test_list = np.load(outfile).tolist()
    print(len(train_list),len(val_list),len(test_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    density_value = args.density_value

    torch.cuda.manual_seed(args.seed)
    task_id = args.task
    model = CSRNet()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    rate_model = RATEnet()
    rate_model = nn.DataParallel(rate_model, device_ids=[0]).cuda()
    ROI_model = []
    criterion = nn.MSELoss(size_average=False).cuda()
    center_cri = CenterLoss(1, 1).cuda()
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': rate_model.parameters(), 'lr':args.rate_lr },
        # {'params': ROI_model.parameters(),'lr':args.lr},
        {'params': center_cri.parameters(), 'lr': lr_cent},
    ], lr=args.lr)

    rate_path = './save_file_ratemodel_ucf/checkpoint.pth.tar'
    #args.pre = './d_model_best_ucf_102.pth.tar'
    args.pre = './refineattention_demo/model_best.pth.tar'
    #args.pre = None
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            
            checkpoint = torch.load(args.pre)

            #checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['pre_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    # Pre_data_train = pre_data(train_list,train=True)
    # Pre_data_val = pre_data(val_list,train=False)
    
    print("best_prec1",best_prec1)
#     for epoch in range(args.start_epoch, args.epochs):
 
    
    for epoch in range( args.start_epoch, 1000):
        adjust_learning_rate(optimizer, epoch)
        start = time.time()
        train(train_list, model, rate_model, ROI_model, criterion, center_cri, optimizer, epoch, density_value, task_id)
        end = time.time()
        prec1, accuracy, original_prec1, visi = validate(val_list, model, rate_model, ROI_model, density_value, task_id)
        end_2 = time.time()
        #print ("train time:",end-start,"test time:",end_2-end)
        is_best = prec1 < best_prec1
        original_is_best = original_prec1 < orginal_best_prec1
        best_prec1 = min(prec1, best_prec1)
        orginal_best_prec1 = min(original_prec1, orginal_best_prec1)
        
        f = open("./interest.bak", 'a+')
        
        if is_best==True:
            test(test_list, model, rate_model, ROI_model, density_value, task_id)
            print(' * is best MAE ', file = f)
        else:
            print(' * MAE {MAE:.3f} '.format(MAE=prec1),
                  ' * best MAE {mae:.3f} '.format(mae=best_prec1),
                  ' * acc {acc:.3f} '.format(acc=accuracy),
                  ' * best original MAE {mae:.3f} '.format(mae=orginal_best_prec1),file = f)
        f.close()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'pre_state_dict': model.state_dict(),
            #'rate_state_dict': rate_model.state_dict(),
            #'roi_state_dict': ROI_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, original_is_best, args.task)
        print(111)


def pre_data(train_list, train):
    print ("Pre_load dataset ......")
    data_keys = {}
    if train:
        train_list = train_list
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, target = load_data(Img_path, train)
        blob = {}
        blob['img'] = img
        blob['gt'] = target
        blob['fname'] = fname
        data_keys[j] = blob
    return data_keys


def target_transform(gt_point, rate):
    point_map = gt_point.cpu().numpy()
    pts = np.array(zip(np.nonzero(point_map)[2], np.nonzero(point_map)[1]))
    pt2d = np.zeros((int(rate * point_map.shape[1]) + 1, int(rate * point_map.shape[2]) + 1), dtype=np.float32)

    for i, pt in enumerate(pts):
        pt2d[int(rate * pt[1]), int(rate * pt[0])] = 1.0

    return pt2d


def gt_transform(pt2d, cropsize, rate):
    [x, y, w, h] = cropsize
    pt2d = pt2d[int(y * rate):int(rate * (y + h)), int(x * rate):int(rate * (x + w))]
    density = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    pts = np.array(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0]))
    orig = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    for i, pt in enumerate(pts):
        orig[int(pt[1]), int(pt[0])] = 1.0

    density += scipy.ndimage.filters.gaussian_filter(orig, 15, mode='constant')

    # print(np.sum(density))
    return density


def stn(theta, x):
    h, w = x.size(2), x.size(3)
    matrix = Variable(torch.Tensor([[[1, 0, 0], [0, 1, 0]]]))

    matrix = matrix.float().cuda()

    matrix = matrix * (1 / theta)
    if theta > 1:
        padding = (
        int(w * (theta - 1) / 2), int(w * (theta - 1) / 2), int(h * (theta - 1) / 2), int(h * (theta - 1) / 2))
        x = F.pad(x, padding, 'constant', 0)

        grid = F.affine_grid(matrix, x.size())
        new = F.grid_sample(x, grid)
        return new
    else:
        grid = F.affine_grid(matrix, x.size())
        new = F.grid_sample(x, grid)
        new = new[:, :, int((1 - theta) * h / 2):h - int((1 - theta) * h / 2),
              int((1 - theta) * w / 2):w - int((1 - theta) * w / 2)]
        return new


def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]

    d1 = d[:, :, abs(int(math.floor((d_h - g_h) / 2.0))):abs(int(math.floor((d_h - g_h) / 2.0))) + g_h,
         abs(int(math.floor((d_w - g_w) / 2.0))):abs(int(math.floor((d_w - g_w) / 2.0))) + g_w]
    return d1


def choose_crop(output, target):
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    return output, target


def resever_img(input_img):

    input_img = torch.cat([torch.cat([input_img[0].squeeze(0), input_img[2].squeeze(0)], 1), torch.cat([input_img[1].squeeze(0), input_img[3].squeeze(0)], 1)], 0).unsqueeze(0).unsqueeze(0)
    #print(input_img.shape, pre_0.shape,pre_up.shape)

    return input_img

def resever_fs(input_fs,train_flag=True):
    # pre_0 = input_fs[0]
    # pre_1 = input_fs[1]
    # pre_2 = input_fs[2]
    # pre_3 = input_fs[3]
    #
    # del input_fs
    # pre_up =
    #
    # del pre_0,pre_1,pre_2,pre_3

    #torch.cuda.empty_cache()
    if train_flag==True:
        input_fs = torch.cat([torch.cat([input_fs[0], input_fs[2]], 2), torch.cat([input_fs[1], input_fs[3]], 2)], 1).unsqueeze(0).cuda(device=2)

    else:
        torch.cuda.empty_cache()
        input_fs = torch.cat([torch.cat([input_fs[0], input_fs[2]], 2), torch.cat([input_fs[1], input_fs[3]], 2)],
                             1).unsqueeze(0)

    #print(input_fs.shape, pre_0.shape,pre_up.shape)

    return input_fs

def detach_img(img,target):
    img = img.squeeze(0)
    target = target.squeeze(0)

    crop_size_x = int(img.shape[1] / 2)
    crop_size_y = int(img.shape[2] / 2)

    x0 = 0
    y0 = 0
    img_return = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    target_return = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)

    x0 = 0 + crop_size_x
    y0 = 0
    img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    img_return = torch.cat([img_return, img_crop], 0)
    target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    target_return = torch.cat([target_return, target_crop], 0)

    x0 = 0
    y0 = 0 + crop_size_y
    img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    img_return = torch.cat([img_return, img_crop], 0)
    target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    target_return = torch.cat([target_return, target_crop], 0)

    x0 = crop_size_x
    y0 = crop_size_y
    img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    img_return = torch.cat([img_return, img_crop], 0)
    target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
    target_return = torch.cat([target_return, target_crop], 0)

    return img_return,target_return


def train(Pre_data, model, rate_model, ROI_model, criterion, center_cri, optimizer, epoch, density_value, task_id):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset_ucf_demo(Pre_data, task_id,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.module.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()
    count_rate = 0
    ori_scale_crop = 0
    target_array_list = []
    density_threshold = 0.002
    count = 0
    sigma_count_list = []
    gt_num_list = []



    for i, (img,fname,target) in enumerate(train_loader):
        # if fname[0]!='img_0375.h5':
        #     continue
        #print(fname)
        
        start = time.time()
        mean_d = []
        loss = 0
        loss_1 = 0
        loss_2 = 0
        c_loss = 0

        data_time.update(time.time() - end)

        img = img.cuda()
        target = target.cuda()
        #print(fname[0], img.shape,target.shape)
        d2, d3, d4, d5, d6 = model(img, target, refine_flag=False)

        target_forward1 = target.unsqueeze(1).type(torch.FloatTensor).cuda()

        #print(img.shape, d6.shape, target.shape, target_forward1.shape,fs.shape)
        loss_2 += criterion(d2, target_forward1) + criterion(d3, target_forward1) + criterion(d4, target_forward1) + criterion(d5, target_forward1) + criterion(d6, target_forward1)


        # d6 = resever_img(d6)
        # img = resever_fs(img)
        #
        # density_map = d6.data.cpu().numpy()
        # [x, y, w, h] = findmaxcontours(density_map, 0, fname)
        #
        # #print(fname)
        # end_1  =  time.time()
        # #print(img.shape, d6.shape, rate_feature.shape,fs.shape)
        # #del    d2, d3, d4, d5, d6
        # if (float(w * h) / (img.size(2) * img.size(3))) > args.threshold and (
        #         float(w * h) / (img.size(2) * img.size(3))) < args.maxthreshold:
        #     fs = resever_fs(fs.cuda(device=1),train_flag = True)
        #     rate_feature = F.adaptive_avg_pool2d(fs[:, :, y:(y + h), x:(x + w)], (14, 14))
        #
        #     kpoint = k[:, y:y + h, x:x + w]
        #     sigma = sigma_map[:, y:y + h, x:x + w]
        #     crop_gt_number = torch.sum(kpoint).cuda()
        #
        #     sigma_count = torch.sum(sigma).type(torch.FloatTensor)
        #
        #
        #     rate = rate_model(rate_feature).clamp_(1, 7)
        #     rate = torch.sqrt(rate)
        #
        #     img_pros = img[:, :, y:(y + h), x:(x + w)]
        #     #          # print(rate)
        #     img_transed = F.upsample_bilinear(img_pros, scale_factor=rate.item())
        #     #img_transed = stn(rate,img_pros)
        #     pt2d = target_transform(k, rate)
        #     target_choose = gt_transform(pt2d, [x, y, w, h], rate.item())
        #     torch.cuda.empty_cache()
        #     target_choose = torch.from_numpy(target_choose).type(torch.FloatTensor).unsqueeze(0).cuda()
        #     #print("refine", rate.item(), img_transed.shape, target_choose.shape)
        #
        #     img_transed, target_choose = detach_img(img_transed,target_choose)
        #     #print("refine", rate.item(), img_transed.shape, target_choose.shape)
        #     dd2, dd3, dd4, dd5, dd6 = model(img_transed, target_choose, refine_flag=False)
        #
        #     count = count + 1
        #     target_choose = target_choose.unsqueeze(1)
        #     #print(dd6.shape, target_choose.shape)
        #     loss_1 += criterion(dd2, target_choose) + criterion(dd3, target_choose) + criterion(dd4,target_choose) + criterion(dd5, target_choose) + criterion(dd6, target_choose)
        #     #print(sigma_count, crop_gt_number,loss_1,loss_2)
        #
        #     if crop_gt_number!=0:
        #         mean_d.append(sigma_count.cuda() / crop_gt_number.type(torch.FloatTensor).cuda() * rate)
        #         center_sample = torch.stack(mean_d, 0).view(len(mean_d), -1)
        #         center_label = torch.zeros(len(mean_d)).type(torch.FloatTensor).cuda()
        #         c_loss += center_cri(center_sample, center_label)
        #     else:
        #         print(fname[0]," gt_num is zero")
        # end_2 = time.time()

        loss += loss_2
        #print(fname[0], "rate:%.3f"%(rate.item()),"space :%.3f"%(float(w * h) / (img.size(2) * img.size(3))), "c_loss: %.3f"% c_loss,"loss:%.3f"%loss)


        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end_3 = time.time()
        f = open("./interest.bak", 'a+')
        if i % args.print_freq == 0:
            f = open("./interest.bak", 'a+')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses),file=f)
            
        end = time.time()
        f.close()
    print("refine number:",count)
        #print(end-start ,end_1-start,end_2-end_1,end_3-end_2)


def validate(Pre_data, model, rate_model, ROI_model, density_value, task_id):
    print ('begin test')
    # Pre_data = pre_data(val_list,train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_ucf_demo(Pre_data, task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse =0
    original_mae = 0
    visi = []
    density_threshold = 0.0005
    start = time.time()
    image_list  = []
    res = []
    gt_res = []
    accuracy = 0
    f = open("./interest.bak", 'a+')
        
    for i, (img,fname,target) in enumerate(test_loader):
#         if i<800:
#             continue
        img = img.cuda()
        target = target.cuda()
        #print (fname[0],img.shape,target.shape)
        d2, d3, d4, d5, d6 = model(img, target, refine_flag=False)
        original_density = d6
        #print(img.shape, d6.shape)
        original_mae += abs(torch.sum(original_density).item()-torch.sum(target).item())
        if torch.sum(target).item()==0:
            print(i,fname)
        else:
            accuracy +=  abs(torch.sum(original_density).item()-torch.sum(target).item())/torch.sum(target).item()
        #print(original_mae)
        image_list.append(fname[0])
        res.append(torch.sum(original_density).item())
        gt_res.append(torch.sum(target).item())
        
        if i % 32 == 0:
            visi.append([img.data.cpu().numpy(), original_density.data.cpu().numpy(),
                         target.unsqueeze(0).data.cpu().numpy(), fname])
            print(i, fname[0],"gt", torch.sum(target).item(),"pred:%.3f"%torch.sum(original_density).item())
        
#         print(gtcount)
#         density_map = original_density.data.cpu().numpy()
#         density_map = 255 * density_map / np.max(density_map)
#         density_map = density_map[0][0]
#         density_map[density_map<0]=0
#         density_map = cv2.applyColorMap(density_map.astype(np.uint8), 2)

#         cv2.imwrite("./video_densitymap/"+fname[0], density_map)
        
    end = time.time()
    mae =original_mae/len(test_loader)
    accuracy = accuracy/(len(test_loader))
    with open('pred_result.txt', 'w') as f:
        for i in range(len(res)):
            f.write(str(res[i]) + '\n')
            
    with open('gt_result.txt', 'w') as f:
        for i in range(len(res)):
            f.write(str(gt_res[i]) + '\n')
            
            
    print(end-start)
    print('MAE:',mae,"acc",accuracy)
    # print(' * ORI_MAE {mae:.3f} '
    #       .format(mae=original_mae))
    return mae,accuracy, original_mae, visi




def test(Pre_data, model, rate_model, ROI_model, density_value, task_id):
    print ('begin test')
    # Pre_data = pre_data(val_list,train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_ucf_test(Pre_data, task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse =0
    original_mae = 0
    visi = []
    density_threshold = 0.0005
    start = time.time()
    image_list  = []
    res = []
    gt_res = []
    accuracy = 0
    for i, (img,fname,target) in enumerate(test_loader):
#         if i<800:
#             continue
        img = img.cuda()
        #print (fname[0],img.shape,target.shape)
        d2, d3, d4, d5, d6 = model(img, img.squeeze(0), refine_flag=False)
        original_density = d6
        #print(img.shape, d6.shape)
        original_mae = torch.sum(original_density).item()
       
        image_list.append(fname[0])
        res.append(original_mae)

        # if i % 32 == 0:
        #     visi.append([img.data.cpu().numpy(), original_density.data.cpu().numpy(),
        #                  target.unsqueeze(0).data.cpu().numpy(), fname])
#         print(i, fname[0], "pred:%.3f"%torch.sum(original_density).item())
        
#         print(gtcount)
#         density_map = original_density.data.cpu().numpy()
#         density_map = 255 * density_map / np.max(density_map)
#         density_map = density_map[0][0]
#         density_map[density_map<0]=0
#         density_map = cv2.applyColorMap(density_map.astype(np.uint8), 2)

#         cv2.imwrite("./video_densitymap/"+fname[0], density_map)
    end = time.time()
    mae =original_mae/len(test_loader)
    accuracy = accuracy/(len(test_loader))
    with open('crowd_result.txt', 'w') as f:
        for i in range(len(res)):
            f.write(image_list[i] + ' ' + str(res[i]) + '\n')
            
    print(end-start)
    print('MAE:',mae,"acc",accuracy)
    # print(' * ORI_MAE {mae:.3f} '
    #       .format(mae=original_mae))
    return mae, original_mae, visi

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #
    if epoch >80:
        args.lr =  5 * 1e-6
    if epoch >300:
        args.lr =  1e-6
    # if epoch>150:
    #     args.lr = 1e-6
    #
    # args.lr = args.original_lr
    #
    # for i in range(len(args.steps)):
    #
    #     scale = args.scales[i] if i < len(args.scales) else 1
    #
    #
    #     if epoch >= args.steps[i]:
    #         args.lr = args.lr * scale
    #         if epoch == args.steps[i]:
    #             break
    #     else:
    #         break
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()        
