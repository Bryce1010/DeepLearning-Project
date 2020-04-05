import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import albumentations as A
from efficientnet_pytorch import EfficientNet
import gc
import cv2
from tqdm import tqdm
import sklearn.metrics
import json


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCH = 40
TQDM_DISABLE = True

device = torch.device("cuda")



## Load Dataset  
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




## Create Dataset  
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
    A.imgaug.transforms.IAAAffine(shear=20, mode='constant', cval=255, always_apply=True),
    A.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], mask_value=[255, 255, 255], always_apply=True),
    A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
    A.Cutout(num_holes=1, max_h_size=112, max_w_size=112, fill_value=128, always_apply=True),
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


font_dataset = GraphemeDataset(font_data, font_images, train_transform)
valid_dataset = GraphemeDataset(font_data, font_images, valid_transform)



## Create Model   

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
    
backbone = EfficientNet.from_name('efficientnet-b0')
classifier = BengalModel(backbone, hidden_size=1280, class_num=168*11*8).to(device)



## Create Data Loader  
font_sampler = torch.utils.data.RandomSampler(font_dataset, True, int(len(font_dataset))*(EPOCH))
valid_sampler = torch.utils.data.RandomSampler(valid_dataset, True, int(len(valid_dataset))*(EPOCH))



font_loader = torch.utils.data.DataLoader(
    font_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=1, 
    pin_memory=True, 
    drop_last=True, 
    sampler=font_sampler)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
    sampler=valid_sampler)

font_loader_iter = iter(font_loader)
valid_loader_iter = iter(valid_loader)


## Training  

def train_step(model, train_iter, criterion, optimizer, scheduler, device):
    image, label = next(train_iter)
    image = image.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    out = model(image)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss

optimizer = torch.optim.AdamW(classifier.parameters())
classifier_loss = nn.CrossEntropyLoss()


num_step_per_epoch = len(font_loader)//EPOCH
num_valid_step_per_epoch = len(valid_loader)//EPOCH
train_steps = num_step_per_epoch*EPOCH
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)



log = []
best_score = 0.


for epoch in range(EPOCH):
    classifier.train()
    metric = {}
    losses = []
    for i in tqdm(range(num_step_per_epoch), disable=TQDM_DISABLE):
        loss = train_step(classifier,
                  font_loader_iter,
                  classifier_loss,
                  optimizer,
                  scheduler,
                  device)        
        losses.append(loss.item())
    metric['train/loss'] = sum(losses)/len(losses)
    classifier.eval()
    preds = []
    labels = []
    for i in tqdm(range(num_valid_step_per_epoch), disable=TQDM_DISABLE):
        image, label = next(valid_loader_iter)
        image = image.to(device)
        with torch.no_grad():
            out = classifier(image)
            pred = out.argmax(dim=1).cpu().numpy()
        
        preds.append(pred)
        labels.append(label.numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    accuracy = sklearn.metrics.accuracy_score(y_pred=preds, y_true=labels)
    metric['valid/accuracy'] = accuracy
    metric['epoch'] = epoch
    
    log.append(metric)
    
    if accuracy > best_score:
        best_score = accuracy
        torch.save(classifier.state_dict(), 'best.pth')
    torch.save(classifier.state_dict(), 'model.pth')
    with open('log.json', 'w') as fout:
        json.dump(log , fout, indent=4)


