import random
import os
from PIL import Image,ImageFilter,ImageDraw
import scipy.io as io
import scipy
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import time
import json
from scipy.ndimage.filters import gaussian_filter

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth_2')

    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    gt = mat["image_info"][0,0][0,0][0]
    k = np.zeros((img.size[1],img.size[0]))
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.size[1] and int(gt[i][0])<img.size[0]:
            k[int(gt[i][1]),int(gt[i][0])]=1


    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.size[1] and int(gt[i][0]) < img.size[0]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    pts = np.array(zip(np.nonzero(k)[1], np.nonzero(k)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(k.shape, dtype=np.float32)
    # pt2d = np.zeros(k.shape,dtype= np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1]) / 2

        sigma_map[pt[1], pt[0]] = sigma

    if train==True:

        if random.random() > 0.5:
            target = np.fliplr(target)
            k = np.fliplr(k)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            sigma_map = np.fliplr(sigma_map)

        if random.random() > 0.5:
            proportion = random.uniform(0.004, 0.015)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))
            #print("noise")

        img=img.copy()
        target=target.copy()
        sigma_map = sigma_map.copy()
        k = k.copy()
    #if train==True:
    #     crop_size = (img.size[0]/2,img.size[1]/2)
    #     if random.randint(0,9)<= -1:
            
            
    #         dx = int(random.randint(0,1)*img.size[0]*1./2)
    #         dy = int(random.randint(0,1)*img.size[1]*1./2)
    #     else:
    #         dx = int(random.random()*img.size[0]*1./2)
    #         dy = int(random.random()*img.size[1]*1./2)
        
        
        
    #     img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
    #     target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
#         if random.random()>0.6:
 #            target = np.fliplr(target)
  #           img = img.transpose(Image.FLIP_LEFT_RIGHT)
   #          k = np.fliplr(k)
    #     img=img.copy()
     #    target=target.copy()
      #   k=k.copy()
    
        # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    
    
    return img,target,k,sigma_map




def load_ucf_data_extract(img_path, train=True):

    '''pytorch1.0'''
    img_path = img_path.decode()

    #img_path = '/home/cfxu/projects/synchronous/chenfeng_code/UCF-QNRF_ECCV18/resize_data_patch_1024/train/img_1122.jpg'
    img_path = img_path.replace('.h5', '.jpg')
    # print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    #target = np.asarray(gt_file['density'])
    #print(gt_path)
    sigma_map = np.asarray(gt_file['sigma_map'])

    k_path = img_path.replace('.jpg', '.npy')
    #print(img.size, img_path)
    gt = np.load(k_path)


    # if train==True:
    #
    #     #rate = random.uniform(0.8, 1.3)
    #
    #     img = img.resize((int(img.size[0] * rate), int(img.size[1] * rate)), Image.ANTIALIAS)
    #     gt = gt * rate

    k = np.zeros((img.size[1], img.size[0]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.size[1] and int(gt[i][0]) < img.size[0]:
            k[int(gt[i][1]), int(gt[i][0])] = 1


    target = gaussian_filter(k, 4)

    # start  = time.time()
    # if np.sum(k) == 0:
    #     k[0, 0] = 1
    #     print("k is 0")
    # #
    # pts = np.array(zip(np.nonzero(k)[1], np.nonzero(k)[0]))
    # leafsize = 2048
    # # build kdtree
    #
    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # # query kdtree
    # distances, locations = tree.query(pts, k=2)
    # sigma_map = np.zeros(k.shape, dtype=np.float32)
    # # pt2d = np.zeros(k.shape,dtype= np.float32)
    # for i, pt in enumerate(pts):
    #     sigma = (distances[i][1]) / 2
    #
    #     sigma_map[pt[1], pt[0]] = sigma

    if train == True:

        # if random.random() > 0.5:
        #     target = np.fliplr(target)
        #     k = np.fliplr(k)
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     sigma_map = np.fliplr(sigma_map)
        #
        # if random.random() > 0.5:
        #     proportion = random.uniform(0.004, 0.01)
        #     width, height = img.size[0], img.size[1]
        #     num = int(height * width * proportion)
        #     for i in range(num):
        #         w = random.randint(0, width - 1)
        #         h = random.randint(0, height - 1)
        #         if random.randint(0, 1) == 0:
        #             img.putpixel((w, h), (0, 0, 0))
        #         else:
        #             img.putpixel((w, h), (255, 255, 255))


        img = img.copy()
        target=target.copy()
        #sigma_map = sigma_map.copy()
        k = k.copy()

    #img.save('1.jpg')
    #target = cv2.cvtColor(target,cv2.COLOR_GRAY2BGR)
    #target = cv2.applyColorMap(target,2)
    #density_map = target / np.max(target) * 255

    #cv2.imwrite("1_ucf.jpg",density_map)
    # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # target  = []

    #print(img.size, target.size,k.size,img_path)
    return img, target, sigma_map, k



def load_ucf_data_k6(img_path, train=True):

    '''pytorch1.0'''
    img_path = img_path.decode()

    #img_path = '/home/cfxu/projects/synchronous/chenfeng_code/UCF-QNRF_ECCV18/resize_data_patch_1024/train/img_1122.jpg'
    img_path = img_path.replace('.h5', '.jpg')
    # print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    #target = np.asarray(gt_file['density'])
    #print(gt_path)
    sigma_map = np.asarray(gt_file['sigma_map'])

    k_path = img_path.replace('.jpg', '.npy')
    #print(img.size, img_path)
    gt = np.load(k_path)


    # if train==True:
    #
    #     rate = random.uniform(0.9, 1.1)
    #
    #     img = img.resize((int(img.size[0] * rate), int(img.size[1] * rate)), Image.ANTIALIAS)
    #     gt = gt * rate

    k = np.zeros((img.size[1], img.size[0]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.size[1] and int(gt[i][0]) < img.size[0]:
            k[int(gt[i][1]), int(gt[i][0])] = 1


    target = gaussian_filter(k, 6)

    # start  = time.time()
    # if np.sum(k) == 0:
    #     k[0, 0] = 1
    #     print("k is 0")
    # #
    # pts = np.array(zip(np.nonzero(k)[1], np.nonzero(k)[0]))
    # leafsize = 2048
    # # build kdtree
    #
    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # # query kdtree
    # distances, locations = tree.query(pts, k=2)
    # sigma_map = np.zeros(k.shape, dtype=np.float32)
    # # pt2d = np.zeros(k.shape,dtype= np.float32)
    # for i, pt in enumerate(pts):
    #     sigma = (distances[i][1]) / 2
    #
    #     sigma_map[pt[1], pt[0]] = sigma

    if train == True:

        if random.random() > 0.5:
            target = np.fliplr(target)
            k = np.fliplr(k)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            sigma_map = np.fliplr(sigma_map)

        if random.random() > 0.5:
            proportion = random.uniform(0.004, 0.01)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))


        img = img.copy()
        target=target.copy()
        sigma_map = sigma_map.copy()
        k = k.copy()

    #img.save('1.jpg')
    #target = cv2.cvtColor(target,cv2.COLOR_GRAY2BGR)
    #target = cv2.applyColorMap(target,2)
    #density_map = target / np.max(target) * 255

    #cv2.imwrite("1_ucf.jpg",density_map)
    # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # target  = []
    #sigma_map = []
    #print(img.size, target.size,k.size,img_path)
    return img, target, sigma_map, k


def load_ucf_data_k8(img_path, train=True):

    '''pytorch1.0'''
    img_path = img_path

    #img_path = '/home/cfxu/projects/synchronous/chenfeng_code/UCF-QNRF_ECCV18/resize_data_patch_1024/train/img_1122.jpg'
    img_path = img_path.replace('.h5', '.jpg')
    # print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    #target = np.asarray(gt_file['density'])
    #print(gt_path)
    #sigma_map = np.asarray(gt_file['sigma_map'])

    k_path = img_path.replace('.jpg', '.npy')
    #print(img.size, img_path)
    gt = np.load(k_path)


    if train==True:

        rate = random.uniform(0.9, 1.1)

        img = img.resize((int(img.size[0] * rate), int(img.size[1] * rate)), Image.ANTIALIAS)
        gt = gt * rate

    k = np.zeros((img.size[1], img.size[0]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.size[1] and int(gt[i][0]) < img.size[0]:
            k[int(gt[i][1]), int(gt[i][0])] = 1


    target = gaussian_filter(k, 8)

    # start  = time.time()
    # if np.sum(k) == 0:
    #     k[0, 0] = 1
    #     print("k is 0")
    # #
    # pts = np.array(zip(np.nonzero(k)[1], np.nonzero(k)[0]))
    # leafsize = 2048
    # # build kdtree
    #
    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # # query kdtree
    # distances, locations = tree.query(pts, k=2)
    # sigma_map = np.zeros(k.shape, dtype=np.float32)
    # # pt2d = np.zeros(k.shape,dtype= np.float32)
    # for i, pt in enumerate(pts):
    #     sigma = (distances[i][1]) / 2
    #
    #     sigma_map[pt[1], pt[0]] = sigma

    if train == True:

        if random.random() > 0.5:
            target = np.fliplr(target)
            k = np.fliplr(k)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #sigma_map = np.fliplr(sigma_map)

        if random.random() > 0.5:
            proportion = random.uniform(0.004, 0.01)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))


        img = img.copy()
        target=target.copy()
        #sigma_map = sigma_map.copy()
        k = k.copy()

    #img.save('1.jpg')
    #target = cv2.cvtColor(target,cv2.COLOR_GRAY2BGR)
    #target = cv2.applyColorMap(target,2)
    #density_map = target / np.max(target) * 255

    #cv2.imwrite("1_ucf.jpg",density_map)
    # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # target  = []
    sigma_map = []
    #print(img.size, target.size,k.size,img_path)
    return img, target, sigma_map, k

def load_ucf_data_demo(img_path, train=True):

    '''pytorch1.0'''

    #img_path = '/home/cfxu/projects/synchronous/chenfeng_code/UCF-QNRF_ECCV18/resize_data_patch_1024/train/img_1122.jpg'

    Img = Image.open(img_path).convert('RGB')
    #print(img_path)
    rate = 1
    if Img.size[0]>=Img.size[1] and Img.size[0]>=1024:
        width = 1024
        height = int(Img.size[1]*1024/Img.size[0])
        rate = 1024.0/Img.size[0]
        Img = Img.resize((width, height),Image.ANTIALIAS)
    if Img.size[1]>Img.size[0] and Img.size[1]>=1024:
        height = 1024
        width = int(Img.size[0]*1024/Img.size[1])
        rate = 1024/Img.size[1]
        Img = Img.resize((width, height),Image.ANTIALIAS)
        #print(img_path)
        #print(width,height)
    
    if train==True:
        rate_random = random.uniform(0.8, 1.3)
        Img = Img.resize((int(Img.size[0] * rate_random), int(Img.size[1] * rate_random)), Image.ANTIALIAS)
        rate = rate * rate_random
    
    points = load_json(img_path, rate)
    
    density_map = np.zeros((Img.size[1], Img.size[0]))
    #print(points[0][1],points)
    for k in range(0, len(points)):
        if (points[k][1]) < Img.size[1] and int(points[k][0]) < Img.size[0]:
            density_map[int(points[k][1]), int(points[k][0])] = 1
    
    density_map = gaussian_filter(density_map, 8)
    target = density_map.copy()
    
#     density_map = density_map / np.max(density_map) * 255
#     density_map = density_map.astype(np.uint8)

#     density_map = cv2.cvtColor(density_map,cv2.COLOR_GRAY2BGR)
#     density_map = cv2.applyColorMap(density_map,2)
#     #print(img_path.split('/'))
#     cv2.imwrite('./visual_densitymap/' + img_path.split('/')[5] + '_demo.jpg', density_map)
# 		# cv2.imwrite('./competition/' + i.split('.')[0] + 'change.jpg', Img)
#     Img.save('./visual_densitymap/' + img_path.split('/')[5] + 'change.jpg')

#     print(img.size)
    if train == True:
        
        if random.random() > 0.5:
            target = np.fliplr(target)

            Img = Img.transpose(Image.FLIP_LEFT_RIGHT)
            #sigma_map = np.fliplr(sigma_map)

        if random.random() > 0.5:
            proportion = random.uniform(0.001, 0.01)
            width, height = Img.size[0], Img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    Img.putpixel((w, h), (0, 0, 0))
                else:
                    Img.putpixel((w, h), (255, 255, 255))
                    
        Img = Img.copy()
        target = target.copy()


    #img.save('1.jpg')
    #target = cv2.cvtColor(target,cv2.COLOR_GRAY2BGR)
    #target = cv2.applyColorMap(target,2)
    #density_map = target / np.max(target) * 255

    #cv2.imwrite("1_ucf.jpg",density_map)
    # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # target  = []
    #sigma_map = []
    #print(img.size, target.size,k.size,img_path)
    return Img,target

def load_json(img_path,rate):
    path = img_path.split('.')[0] +"."+ img_path.split('.')[1]  + str('.json')
    with open(path,'r') as f:
        j = json.loads(f.read())
        points = []
        shape_list = j['shapes']
		#print('./competition/' + i.replace('json','jpg'))
        for shape in shape_list:
            #print(shape['label'],shape['shape_type'])
            if shape['label'] != 'people':
                #print('not people')
                continue
            if shape['shape_type'] == 'rectangle':
				#                 print('rec',shape['points'])
               
                [x1, y1], [x2, y2] = shape['points']
                center_points = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        
            elif shape['shape_type'] == 'point':
				#                 print('point',shape['points'])
                [center_points] = shape['points']
            #print(center_points,rate)    
            
            center_points[0] = center_points[0]*rate
            center_points[1] = center_points[1]*rate
            #print(center_points)
            points.append(center_points)
        
        #print(img_path,"points",points,len(shape_list))
    return points



def load_ucf_data_test(img_path, train=True):

    '''pytorch1.0'''

    #img_path = '/home/cfxu/projects/synchronous/chenfeng_code/UCF-QNRF_ECCV18/resize_data_patch_1024/train/img_1122.jpg'

    Img = Image.open(img_path).convert('RGB')
    #print(img_path)
    if Img.size[0]>=Img.size[1] and Img.size[0]>=1024:
        width = 1024
        height = int(Img.size[1]*1024/Img.size[0])
        rate = 1024.0/Img.size[0]
        Img = Img.resize((width, height),Image.ANTIALIAS)
    if Img.size[1]>Img.size[0] and Img.size[1]>=1024:
        height = 1024
        width = int(Img.size[0]*1024/Img.size[1])
        rate = 1024/Img.size[1]
        Img = Img.resize((width, height),Image.ANTIALIAS)

#     print(img.size)
    if train == True:
        Img = Img.copy()
        target = []

    target = []
    #img.save('1.jpg')
    #target = cv2.cvtColor(target,cv2.COLOR_GRAY2BGR)
    #target = cv2.applyColorMap(target,2)
    #density_map = target / np.max(target) * 255

    #cv2.imwrite("1_ucf.jpg",density_map)
    # print(img.shape)
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # target  = []
    #sigma_map = []
    #print(img.size, target.size,k.size,img_path)
    return Img,target