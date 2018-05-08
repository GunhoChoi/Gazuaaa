import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7" #
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import scipy.io as io
from data.augmentation import *
from data.dataloader import *

from models.vgg_3d import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable


train_dir = '../dataset/New_Bacteria/7_class/train'
test_dir  = '../dataset/New_Bacteria/7_class/test'

#model_dir = '/home/ubuntu/Desktop/Tomocube/code/gunho/first/save/highest_model_[85.].pkl'

num_gpu = 8

n_classes = 7
num_epoch = 10000
batch_size = 30 * num_gpu
test_batch_size = 14 * num_gpu
lr = 0.0002


augmentation_list = [preprocess_crop_3d,
                     crop_3d,
                     gaussian_3d,
                     calibration,
                     elastic_transform,
                     flipud_3d,
                     fliplr_3d,
                     rotate_3d,
                     to_tensor]

test_augmentation = [preprocess_crop_3d,
                     center_crop_3d,
                     calibration,
                     to_tensor]


train_set = ImageFolder(train_dir,transform=augmentation_list)
train_batch = data.DataLoader(train_set,batch_size=batch_size, shuffle=True, drop_last=True)

test_set = ImageFolder(test_dir,transform=test_augmentation)
test_batch = data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True, drop_last=True)


print(train_set.class_to_idx,train_set.__len__())
print(test_set.class_to_idx,test_set.__len__())


# VGG-19
model = vgg19_bn(num_classes=n_classes)

# Attention VGG-19

model = nn.DataParallel(model.cuda())

try:
    model.load_state_dict(torch.load(model_dir))
    print("model restored")
except:
    print("model not restored")


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)


highest_accuracy = torch.FloatTensor([0])
for i in range(num_epoch):
    print("\n----------{}th epoch starting----------\n".format(i))
    
    model.train()
    top_1_count = torch.FloatTensor([0])
    for train_idx,(img,label) in enumerate(train_batch):
        #print("It's working")
        optimizer.zero_grad()

        x = Variable(img.type_as(torch.FloatTensor()))
        y = Variable(label.cuda())
        out = model(x)
          
        loss = loss_function(out,y)
        loss.backward()
        optimizer.step()
        
        # accuracy

        values, idx = out.max(dim=1)
        top_1_count += torch.sum(y==idx).float().cpu().data
        
    #accuracy = 100*top_1_count.cpu()/(train_set.__len__())
    print((train_idx+1)*batch_size)
    train_accuracy = 100*top_1_count.cpu()/((train_idx+1)*batch_size)
    print("train accuracy: {}%".format(train_accuracy.numpy()))
	

    # print("\n----------start Testing----------\n")
    model.eval()
    top_1_count = torch.FloatTensor([0])
    for test_idx,(image,label) in enumerate(test_batch):

        x = Variable(image.type_as(torch.FloatTensor()),volatile=True).cuda()
        y = Variable(label.cuda())
        
        output = model(x)

        # accuracy

        values,idx = output.max(dim=1)
        top_1_count += torch.sum(y==idx).float().cpu().data


    #accuracy = 100*top_1_count.cpu()/(test_set.__len__())
    print((test_idx+1)*test_batch_size)
    test_accuracy = 100*top_1_count.cpu()/((test_idx+1)*test_batch_size)
    print("test accuracy: {}%".format(test_accuracy.numpy()))
    #torch.save(model.state_dict(),"./save/7_class_last_model.pkl")
    
    total_accuracy = train_accuracy + test_accuracy
    if total_accuracy.numpy() > highest_accuracy.numpy():
        highest_accuracy = total_accuracy
        torch.save(model.state_dict(),"./save/vgg_7_class_highest_model_{}_{}.pkl".format(train_accuracy.numpy(),test_accuracy.numpy()))
        print("train accuracy: {}%  test accuracy: {}%".format(train_accuracy.numpy(),test_accuracy.numpy()))
        print("highest model saved")
        
    print("highest accuracy: {}% ".format(highest_accuracy.numpy()))