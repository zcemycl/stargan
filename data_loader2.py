from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from skimage import io,transform

def filterStrings(target,rmvs=['\n']):
    target = target.strip()
    for i in rmvs:
        target = target.replace(i,'')
    return target

def readBbox(f):
    lines = f.readlines()
    noimg = int(lines[0].replace('\n',''))
    imgbbox = lines[2:]
    jpgid,boxprop = [],[]
    for imgLine in imgbbox:
        imgItems = imgLine.replace('\n','').split(' ')
        imgItems = list(map(filterStrings,imgItems))
        imgItems = [item for item in imgItems if item != '']
        jpgid.append(imgItems[0])
        boxprop.append(list(map(int,imgItems[1:])))
    return noimg,jpgid,boxprop

def readLandmarks(f):
    lines = f.readlines()
    noimg = int(lines[0].replace('\n',''))
    landmarksList = lines[1].replace('\n','').split(' ')
    imgmarks = lines[2:]
    jpgid,lmList = [],[]
    for imgLine in imgmarks:
        imgItems = imgLine.replace('\n','').split(' ')
        imgItems = list(map(filterStrings,imgItems))
        imgItems = [item for item in imgItems if item != '']
        jpgid.append(imgItems[0])
        lmList.append(list(map(int,imgItems[1:])))
    return noimg,jpgid,lmList

def refinebbox(image,bbox,lmk,mode='b'):
    if mode=='l':
        dimx,dimy = image.shape[:2]
        xs,ys = lmk[::2],lmk[1::2]
        width = max(xs[1]-xs[0],xs[4]-xs[3])
        height = -min(ys[0],ys[1])+max(ys[3],ys[4])
        miny = max(0,ys[2]-3*height)
        maxy = min(dimy,ys[2]+3*height)
        minx = max(0,xs[2]-3*width)
        maxx = min(dimx,xs[2]+3*width)
        sample = image[miny:maxy,minx:maxx,:]
    elif mode=='b':
        x,y,w,h = bbox
        sample = image[y:y+h,x:x+w,:]
    return sample

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.bboxfile = open('/media/yui/Disk/data/CelebA/Anno/list_bbox_celeba.txt','r')
        self.lmsfile = open('/media/yui/Disk/data/CelebA/Anno/list_landmarks_celeba.txt','r')
        _,self.jpgIDs,self.boxprop = readBbox(self.bboxfile)
        _,self.jpgids,self.lmList = readLandmarks(self.lmsfile)
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = io.imread(os.path.join(self.image_dir,filename))
        #print(filename,self.jpgids[int(filename.replace('.jpg',''))-1])
        lmk = self.lmList[int(filename.replace('.jpg',''))-1]
        bbox = self.boxprop[int(filename.replace('.jpg',''))-1]
        sample = refinebbox(image,bbox,lmk,mode='l')
        cropimg = Image.fromarray(sample)
        return self.transform(cropimg), torch.FloatTensor(label)
        #image = Image.open(os.path.join(self.image_dir, filename))
        #return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize((image_size,image_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
