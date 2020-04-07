from data_loader2 import *
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="test data_loader2")
parser.add_argument('--image_dir',type=str,default='/media/yui/Disk/data/CelebA/img_celeba')
parser.add_argument('--attr_path',type=str,default='/media/yui/Disk/data/CelebA/Anno/list_attr_celeba.txt')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--crop_size',type=int,default=178)
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--image_size',type=int,default=128)
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--num_workers',type=int,default=1)
args = parser.parse_args()
transform = []
if args.mode == 'train':
    transform.append(T.RandomHorizontalFlip())
transform.append(T.Resize((args.image_size,args.image_size)))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

dataset = CelebA(args.image_dir,args.attr_path,args.selected_attrs,transform,args.mode)
print(dataset[0][0].size())
plt.imshow(np.transpose(dataset[0][0].numpy(),(1,2,0)))
plt.show()
data_loader = data.DataLoader(dataset=dataset,
        batch_size=args.batch_size,shuffle=(args.mode=='train'),
        num_workers=args.num_workers)
