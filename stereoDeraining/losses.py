import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from train_stereo import opt


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        #fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, image):

        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


class VGG_Loss(nn.Module):
    def __init__(self):
        super(VGG_Loss, self).__init__()
        self.opt = opt
        self.gpu_id = opt.gpu_id
        #self.device = torch.device('cuda:{}'.format(self.gpu_id[0])) if self.gpu_id else torch.device('cpu')
        self.gpu1 = torch.device('cuda:0')
        self.vgg16_extractor = VGG16FeatureExtractor().to(self.gpu1)
        self.vgg16_extractor = torch.nn.DataParallel(self.vgg16_extractor, [0])

        self.criterionL2_style_loss = torch.nn.MSELoss()
        self.criterionL2_content_loss = torch.nn.MSELoss()

    def forward(self, img1, img2):
        vgg_ft_out = self.vgg16_extractor(img1)
        vgg_ft_target = self.vgg16_extractor(img2)
        self.loss_style = 0
        self.loss_content = 0
        for i in range(3):
            self.loss_style += self.criterionL2_style_loss(gram_matrix(vgg_ft_out[i]), gram_matrix(vgg_ft_target[i]))
            self.loss_content += self.criterionL2_content_loss(vgg_ft_out[i], vgg_ft_target[i])

        self.loss_style *= self.opt.style_weight
        self.loss_content *= self.opt.content_weight

        return self.loss_style, self.loss_content


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        bz, _, h, w = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / bz

    @staticmethod
    def _tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)


class Cosine(nn.Module):
    def __init__(self):
        super(Cosine, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        assert x.size() == y.size()
        #assert x2.size() == y2.size()
        batch_size = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_ = x.add(1e-8)
        y_ = y.add(1e-8)
        mul =torch.abs(x_.mul(y_))
        mul_sum = torch.sum(mul,1) # 256*256
        x_norm = torch.norm(x_, p=2, dim = 1)
        y_norm = torch.norm(y_, p=2, dim = 1)
        cosine = 1 - torch.sum(torch.div(mul_sum,(x_norm.mul(y_norm) ))) / (h * w * batch_size)
        #l1 = self.l1(x,y)
        #l1_2 = self.l1(x2,y2)
        # print('l1 = {},{}, cosine = {}'.format(l1,l1_2,cosine))
        #return 5* cosine + l1 + 0.8 * l1_2
        return 5 * cosine

