import torch
import torch.nn as nn
from torch.autograd import Variable

class Attention_Layer_3D(nn.Module):
    def __init__(self,feature_size,global_size,inter_channel):
        super(Attention_Layer_3D,self).__init__()
        self.feature_size = feature_size # batch,channel,x,y,z
        self.global_size = global_size   # batch,channel,x,y,z
        self.inter_channel = inter_channel

        self.feature_layer = nn.Conv3d(self.feature_size[1],self.inter_channel,kernel_size=1)
        self.global_layer = nn.Conv3d(self.global_size[1],self.inter_channel,kernel_size=1,bias=False)
        self.upsample_global = nn.Upsample(size=self.feature_size[2:],mode='trilinear')
        self.relu = nn.ReLU(inplace=True)
        self.to_1_channel = nn.Conv3d(self.inter_channel,1,kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.GAP_layer = nn.AdaptiveAvgPool3d(output_size=(1,1,1))

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

    def normalize_alpha(self,alpha):
        batch,c,x,y,z = alpha.size()
        alpha_2d = alpha.view(batch,c*x*y*z)
        min_val,_ = torch.min(alpha_2d,dim=1)
        min_val = min_val.view(batch,-1)
        alpha_2d = alpha_2d - min_val.expand_as(alpha_2d)

        sum_val = torch.sum(alpha_2d,dim=1).view(batch,-1)
        alpha_2d = alpha_2d/sum_val.expand_as(alpha_2d)
        alpha = alpha_2d.view(batch,c,x,y,z)
        return alpha

    def forward(self,feature_tensor,global_tensor):
        inter_feature = self.feature_layer(feature_tensor)
        inter_global = self.global_layer(global_tensor)
        global_upsample = self.upsample_global(inter_global)

        assert inter_feature.size() == global_upsample.size()
        sum_relu = self.relu(inter_feature+global_upsample)
        compatibility = self.to_1_channel(sum_relu) # batch,1,x,y,z
        batch,c,x,y,z = compatibility.size()
        compatibility = compatibility.view(batch,c,x*y*z)
        
        # alpha calculation
        alpha = self.softmax(compatibility)
        alpha = alpha.view(batch,c,x,y,z)
        normalized_alpha = self.normalize_alpha(alpha)
        expanded_alpha = normalized_alpha.expand(*self.feature_size)

        # weighted feature map calculation
        assert expanded_alpha.size()==feature_tensor.size()
        weighted_feature = expanded_alpha * feature_tensor
        gap_weighted_feature = self.GAP_layer(weighted_feature)

        return normalized_alpha,gap_weighted_feature