import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torchvision
from torchvision import transforms
import albumentations as A
from efficientnet_pytorch import EfficientNet
import gc
import cv2
from tqdm import tqdm
import sklearn.metrics
import json
import functools
import itertools
import random


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
EPOCH = 1
TQDM_DISABLE = True

device = torch.device("cuda")


def load_images(paths):
    all_images = []
    for path in paths:
        image_df = pd.read_parquet(path)
        images = image_df.iloc[:, 1:].values.reshape(-1, 137, 236).astype(np.uint8)
        del image_df
        gc.collect()
        all_images.append(images)
    all_images = np.concatenate(all_images)
    return all_images


font_data = pd.read_csv('../input/bengaliai-cv19-font/font.csv')
font_images = load_images([
    '../input/bengaliai-cv19-font/font_image_data_0.parquet',
    '../input/bengaliai-cv19-font/font_image_data_1.parquet',
    '../input/bengaliai-cv19-font/font_image_data_2.parquet',
    '../input/bengaliai-cv19-font/font_image_data_3.parquet',
])
np.save('font_images.npy', font_images)
del font_images
gc.collect()

train_data = pd.read_csv('../input/bengaliai-cv19/train.csv')
multi_diacritics_train_data = pd.read_csv('../input/bengaliai-cv19/train_multi_diacritics.csv')
train_data = train_data.set_index('image_id')
multi_diacritics_train_data = multi_diacritics_train_data.set_index('image_id')
train_data.update(multi_diacritics_train_data)
train_images = load_images([
    '../input/bengaliai-cv19/train_image_data_0.parquet',
    '../input/bengaliai-cv19/train_image_data_1.parquet',
    '../input/bengaliai-cv19/train_image_data_2.parquet',
    '../input/bengaliai-cv19/train_image_data_3.parquet',
]) 


font_images = np.load('font_images.npy')
# !rm ./font_images.npy
class GraphemeDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, images, transform=None, num_grapheme_root=168, num_vowel_diacritic=11, num_consonant_diacritic=8):
        self.data = data
        self.grapheme_root_list = np.array(data['grapheme_root'].tolist(), dtype=np.int64)
        self.vowel_diacritic_list = np.array(data['vowel_diacritic'].tolist(), dtype=np.int64)
        self.consonant_diacritic_list = np.array(data['consonant_diacritic'].tolist(), dtype=np.int64)
        self.num_grapheme_root = num_grapheme_root
        self.num_vowel_diacritic = num_vowel_diacritic
        self.num_consonant_diacritic = num_consonant_diacritic
        self.images = images
        self.transform = transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        grapheme_root = self.grapheme_root_list[idx]
        vowel_diacritic = self.vowel_diacritic_list[idx]
        consonant_diacritic = self.consonant_diacritic_list[idx]
        label = (grapheme_root*self.num_vowel_diacritic+vowel_diacritic)*self.num_consonant_diacritic+consonant_diacritic
        np_image = self.images[idx].copy()
        out_image = self.transform(np_image)
        return out_image, label


class Albumentations:
    def __init__(self, augmentations):
        self.augmentations = A.Compose(augmentations)
    
    def __call__(self, image):
        image = self.augmentations(image=image)['image']
        return image
        

preprocess = [
    A.CenterCrop(height=137, width=IMG_WIDTH),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
]

augmentations = [
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], always_apply=True),
    A.imgaug.transforms.IAAAffine(shear=5, mode='constant', cval=255, always_apply=True),
    A.ShiftScaleRotate(rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], mask_value=[255, 255, 255], always_apply=True),
    A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
]


train_transform = transforms.Compose([
    np.uint8,
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    np.uint8,
    Albumentations(preprocess + augmentations),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
#     transforms.ToPILImage(),
])
valid_transform = transforms.Compose([
    np.uint8,
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    np.uint8,
    Albumentations(preprocess),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
#     transforms.ToPILImage(),
])


hand_dataset = GraphemeDataset(train_data, train_images, valid_transform)
font_dataset = GraphemeDataset(font_data, font_images, train_transform)




## CycleGan model
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

def init_weight(net, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias'):
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss



class BengalModel(nn.Module):
    def __init__(self, backbone, hidden_size=2560, class_num=168*11*7):
        super(BengalModel, self).__init__()
        self.backbone = backbone
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_size, class_num)
        self.ln = nn.LayerNorm(hidden_size)

        
    def forward(self, inputs):
        bs = inputs.shape[0]
        feature = self.backbone.extract_features(inputs)
        feature_vector = self._avg_pooling(feature)
        feature_vector = feature_vector.view(bs, -1)
        feature_vector = self.ln(feature_vector)

        out = self.fc(feature_vector)
        return out   


norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
generator_a = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
generator_b = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

discriminator_a = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer)
discriminator_b = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer)
backbone = EfficientNet.from_name('efficientnet-b0')
classifier = BengalModel(backbone, hidden_size=1280, class_num=168*11*8)
init_weight(generator_a, 0.02)
init_weight(generator_b, 0.02)
init_weight(discriminator_a, 0.02)
init_weight(discriminator_b, 0.02)



discriminator_loss = GANLoss('lsgan', target_real_label=1.0, target_fake_label=0.0)
classifier_loss = nn.CrossEntropyLoss()



classifier.load_state_dict(torch.load('../input/cyclegan-classifier-results/best.pth'))


class CycleGan(nn.Module):
    
    def __init__(self, 
                 generator_a, generator_b, discriminator_a, discriminator_b, classifier, 
                 discriminator_loss, classifier_loss, 
                 lambda_a, lambda_b, lambda_cls,
                 device):
        super(CycleGan, self).__init__()
        self.generator_a = generator_a
        self.generator_b = generator_b
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b
        self.classifier = classifier.eval()
        CycleGan.set_requires_grad(self.classifier, requires_grad=False)
        self.discriminator_loss = discriminator_loss
        self.classifier_loss = classifier_loss
        self.reconstruct_loss = nn.L1Loss()
        self.device = device
        
        self.image_pool_a = ImagePool(50)
        self.image_pool_b = ImagePool(50)
        
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_cls = lambda_cls
        
        self.real_images_a = None
        self.real_images_b = None
        self.labels_a = None
        self.labels_b = None
        self.fake_images_a = None
        self.fake_images_b = None
        self.rec_images_a = None
        self.rec_images_b = None
        self.generator_a = torch.nn.DataParallel(self.generator_a)
        self.generator_b = torch.nn.DataParallel(self.generator_b)
        self.discriminator_a = torch.nn.DataParallel(self.discriminator_a)
        self.discriminator_b = torch.nn.DataParallel(self.discriminator_b)
        self.to(device)
        
    def forward(self):
        self.fake_images_a = self.generator_a(self.real_images_b)
        self.fake_images_b = self.generator_b(self.real_images_a)
        self.rec_images_a = self.generator_a(self.fake_images_b)
        self.rec_images_b = self.generator_b(self.fake_images_a)
    
        
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
                    
    def generator_step(self):
        CycleGan.set_requires_grad([self.discriminator_a, self.discriminator_b], False)
        
        loss_a = self.discriminator_loss(self.discriminator_a(self.fake_images_a), True)
        loss_b = self.discriminator_loss(self.discriminator_b(self.fake_images_b), True)
        cycle_a = self.reconstruct_loss(self.rec_images_a, self.real_images_a)*self.lambda_a
        cycle_b = self.reconstruct_loss(self.rec_images_b, self.real_images_b)*self.lambda_b
        cls_loss = self.classifier_loss(self.classifier(self.fake_images_b), self.labels_a)*self.lambda_cls
        
        loss = loss_a + loss_b + cycle_a + cycle_b + cls_loss
        loss.backward()
        CycleGan.set_requires_grad([self.discriminator_a, self.discriminator_b], True)
        return loss, loss_a, loss_b, cycle_a, cycle_b, cls_loss
        
    def discriminator_step(self):
        pred_real_a = self.discriminator_a(self.real_images_a)
        loss_real_a = self.discriminator_loss(pred_real_a, True)
        fake_images_a = self.image_pool_a.query(self.fake_images_a).detach()
        pred_fake_a = self.discriminator_a(fake_images_a)
        loss_fake_a = self.discriminator_loss(pred_fake_a, False)
        
        pred_real_b = self.discriminator_b(self.real_images_b)
        loss_real_b = self.discriminator_loss(pred_real_b, True)
        fake_images_b = self.image_pool_b.query(self.fake_images_b).detach()
        pred_fake_b = self.discriminator_b(fake_images_b)
        loss_fake_b = self.discriminator_loss(pred_fake_b, False)
        
        loss = (loss_real_a + loss_fake_a)/2 + (loss_real_b + loss_fake_b)/2
        loss.backward()
        return loss, loss_real_a, loss_fake_a, (loss_real_a + loss_fake_a)/2, loss_real_b, loss_fake_b, (loss_real_b+loss_fake_b)/2
    
    def set_input(self, images_a, images_b, labels_a, labels_b):
        self.real_images_a = images_a.to(self.device)
        self.real_images_b = images_b.to(self.device)
        self.labels_a = labels_a
        self.labels_b = labels_b


model = CycleGan(generator_a=generator_a,
                generator_b=generator_b,
                discriminator_a=discriminator_a,
                discriminator_b=discriminator_b,
                classifier=classifier,
                discriminator_loss=discriminator_loss,
                classifier_loss=classifier_loss,
                lambda_a=10.0,
                lambda_b=10.0,
                lambda_cls=1.0,
                device=device
                )


hand_sampler = torch.utils.data.RandomSampler(hand_dataset, True, int(max(len(hand_dataset), len(font_dataset)))*(EPOCH))
font_sampler = torch.utils.data.RandomSampler(font_dataset, True, int(max(len(hand_dataset), len(font_dataset)))*(EPOCH))
hand_loader = torch.utils.data.DataLoader(
    hand_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=1, 
    pin_memory=True, 
    drop_last=True, 
    sampler=hand_sampler)
font_loader = torch.utils.data.DataLoader(
    font_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=1, 
    pin_memory=True, 
    drop_last=True, 
    sampler=font_sampler)



hand_loader_iter = iter(hand_loader)
font_loader_iter = iter(font_loader)


def train_step(model, a_iter, b_iter, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, device):
    a_image, a_label = next(a_iter)
    b_image, b_label = next(b_iter)
    a_image = a_image.to(device)
    b_image = b_image.to(device)
    a_label = a_label.to(device)
    b_label = b_label.to(device)
    model.set_input(a_image, b_image, a_label, b_label)
    model.forward()
    generator_optimizer.zero_grad()
    generator_loss, generator_loss_a, generator_loss_b, cycle_a, cycle_b, cls_loss = model.generator_step()
    generator_optimizer.step()
    discriminator_optimizer.zero_grad()
    discriminator_loss, loss_real_a, loss_fake_a, discriminator_loss_a, loss_real_b, loss_fake_b, discriminator_loss_b = model.discriminator_step()
    discriminator_optimizer.step()
    generator_scheduler.step()
    discriminator_scheduler.step()
    return generator_loss, generator_loss_a, generator_loss_b, cycle_a, cycle_b, cls_loss, discriminator_loss, loss_real_a, loss_fake_a, discriminator_loss_a, loss_real_b, loss_fake_b, discriminator_loss_b






generator_optimizer = torch.optim.Adam(itertools.chain(generator_a.parameters(), generator_b.parameters()), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(itertools.chain(discriminator_a.parameters(), discriminator_b.parameters()), lr=0.0002, betas=(0.5, 0.999))



num_step_per_epoch = len(hand_loader)//EPOCH
train_steps = num_step_per_epoch*EPOCH
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, warmup_linear_decay)
discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, warmup_linear_decay)



class LossAverager:
    def __init__(self, prefix):
        self.prefix = prefix
        self.generator_loss = []
        self.generator_loss_a = []
        self.generator_loss_b = []
        self.cycle_a = []
        self.cycle_b = []
        self.cls_loss = []
        self.discriminator_loss = []
        self.loss_real_a = []
        self.loss_fake_a = []
        self.discriminator_loss_a = []
        self.loss_real_b = []
        self.loss_fake_b = []
        self.discriminator_loss_b = []
    
    def append(self, generator_loss, generator_loss_a, generator_loss_b, cycle_a, cycle_b, cls_loss, discriminator_loss, loss_real_a, loss_fake_a, discriminator_loss_a, loss_real_b, loss_fake_b, discriminator_loss_b):
        self.generator_loss.append(generator_loss.item())
        self.generator_loss_a.append(generator_loss_a.item())
        self.generator_loss_b.append(generator_loss_b.item())
        self.cycle_a.append(cycle_a.item())
        self.cycle_b.append(cycle_b.item())
        self.cls_loss.append(cls_loss.item())
        self.discriminator_loss.append(discriminator_loss.item())
        self.loss_real_a.append(loss_real_a.item())
        self.loss_fake_a.append(loss_fake_a.item())
        self.discriminator_loss_a.append(discriminator_loss_a.item())
        self.loss_real_b.append(loss_real_b.item())
        self.loss_fake_b.append(loss_fake_b.item())
        self.discriminator_loss_b.append(discriminator_loss_b.item())

    def average(self):
        metric = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                metric[self.prefix+'/'+key] = sum(value)/len(value)
        return metric


log = []


for epoch in range(EPOCH):
    model.train()
    model.classifier.eval()
    loss_averager = LossAverager('train')
    for i in tqdm(range(num_step_per_epoch)):
        losses = train_step(model, hand_loader_iter, font_loader_iter, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, device)
        loss_averager.append(*losses)
    metric = loss_averager.average()
    metric['epoch'] = epoch
    model.eval()
    log.append(metric)
    torch.save(generator_b.state_dict(), 'generator.pth')
    with open('log.json', 'w') as fout:
        json.dump(log , fout, indent=4)
    




