import argparse
from imutils import paths
from model import Generator
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from skimage import io,transform
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test Generator Model')
parser.add_argument('--test-images-dir',type=str,default='/home/yui/Documents/projects/cv/stargan/test/images')
parser.add_argument('--test-results-dir',type=str,default='/home/yui/Documents/projects/cv/stargan/test/results')
parser.add_argument('--generator-weight',type=str,default='/home/yui/Documents/projects/cv/stargan/stargan_celeba4/models/200000-G.ckpt')
parser.add_argument('--set-attributes',nargs='+')
args = parser.parse_args()

class testDataset(data.Dataset):
    def __init__(self,img_dir,set_attrs,transform):
        self.image_dir = img_dir
        self.imagePaths = list(paths.list_images(self.image_dir))
        self.set_attrs = set_attrs
        self.transform = transform
    def __getitem__(self,idx):
        image = io.imread(self.imagePaths[idx])
        image = Image.fromarray(image)
        label = list(map(float,self.set_attrs))
        return self.transform(image),torch.FloatTensor(label)
    def __len__(self):
        return len(self.imagePaths)

def loadtransform(image_size=128):
    transform = []
    transform.append(T.Resize((image_size,image_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform

def loadGenerator():
    generator = Generator()
    generator.load_state_dict(torch.load(args.generator_weight))
    device = torch.device("cuda") 
    generator.to(device)
    return generator

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def main():
    generator = loadGenerator()
    transform = loadtransform()
    dataset = testDataset(args.test_images_dir,args.set_attributes,transform)
    data_loader = data.DataLoader(dataset=dataset,batch_size=16,
            shuffle=False,num_workers=1)
    for (i,(image,label)) in enumerate(data_loader):
        with torch.no_grad():
            result = generator(image.cuda(),label.cuda())   
        result = denorm(result.data.cpu()).numpy().squeeze()
        result = np.transpose(result,(1,2,0))
        print(result.shape)
        print(np.max(result),np.min(result))
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()
