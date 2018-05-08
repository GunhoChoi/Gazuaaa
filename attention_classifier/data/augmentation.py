import torch
import random
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

'''
def preprocess_crop_3d(data,new_size=[144,144,80]):
    img = data["data"]
    origin_size = img.shape
    data["data"] = img[78:222,78:222,20:-21]
    return data


def crop_3d(img,target_shape= [96,96,80]):
    origin_size = img["data"].shape
    rand_1 = random.randint(0,origin_size[0]-target_shape[0])
    rand_2 = random.randint(0,origin_size[1]-target_shape[1])
    img["data"] = img["data"][rand_1:rand_1+target_shape[0],rand_2:rand_2+target_shape[1],:]
    return img


def center_crop_3d(img,target_shape= [96,96,80]):
    img["data"] = img["data"][24:-24,24:-24,:]
    return img
'''

def preprocess_crop_3d(data,new_size=[144,144,64]):
    img = data["data"]
    origin_size = img.shape
    data["data"] = img[78:222,78:222,28:92]
    return data


def crop_3d(img,target_shape= [96,96,64]):
    origin_size = img["data"].shape
    rand_1 = random.randint(0,origin_size[0]-target_shape[0])
    rand_2 = random.randint(0,origin_size[1]-target_shape[1])
    img["data"] = img["data"][rand_1:rand_1+target_shape[0],rand_2:rand_2+target_shape[1],:]
    return img


def center_crop_3d(img,target_shape= [96,96,64]):
    img["data"] = img["data"][24:-24,24:-24,:]
    return img


def gaussian_3d(img):
    size = img["data"].shape
    sigma = random.uniform(0.01,0.05)
    noise = np.random.normal(0,sigma,size=size)
    img["data"] = img["data"]+noise
    return img


def calibration(img):
    data = img["data"]
    ri = img["ri"]
    img_calibrated = (data - ri) * 9
    img_calibrated[img_calibrated < 0] = 0
    img["data"] = img_calibrated
    return img


def no_calibration(img):
    return img


def flipud_3d(img):
    data = img["data"]
    rand = random.randint(0,1)
    if rand==0:
        img["data"] = data[::-1,:,:].copy()
        return img
    else: 
        return img


def fliplr_3d(img):
    data = img["data"]
    rand = random.randint(0,1)
    if rand ==0:
        img["data"]= data[:,::-1,:].copy()
        return img
    else:
        return img


# https://github.com/scipy/scipy/issues/5925
def rotate_3d(img):
    data = img["data"]
    #rand = random.randint(1,360)
    rand = random.randrange(0,360,45)
    data = ndimage.interpolation.rotate(data,rand,reshape=False,order=0,mode='reflect')
    img["data"] = data
    return img


def to_tensor(img):
    img = img["data"]
    img = torch.from_numpy(img).unsqueeze(dim=0)
    return img


# (1, 1), (5, 2), (1, 0.5), (1, 3) 
def elastic_transform(img, alpha=0, sigma=0, random_state=None):
    '''
	Elastic deformation of images as described in [Simard2003]_.
	.. [Simard2003] Simard, Steinkraus and Platt, “Best Practices for
	Convolutional Neural Networks applied to Visual Document Analysis”, in
	Proc. of the International Conference on Document Analysis and
	Recognition, 2003.
	'''
    data = img["data"]
    param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    rand = random.randint(0,3)
    alpha,sigma = param_list[rand]
    
    #alpah = [1,5], sigma =[0.5,3]
    #alpha = random.uniform(1,1)
    #sigma = random.uniform(1,3)


    assert len(data.shape)==3
    if random_state is None:
       random_state = np.random.RandomState(None)    

    shape = data.shape[0:2]    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    new = np.zeros(data.shape)
    for i in range(data.shape[2]):
        new[:, :, i] = map_coordinates(data[:, :, i], indices, order=1).reshape(shape)
    
    img["data"] = new
    return img