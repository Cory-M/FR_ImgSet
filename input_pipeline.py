import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np
import torch
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

#获取路径下的每段视频的路径及label
#返回一个列表，每一个元素是(视频地址,视频标签)
def make_video_list(dir,class_to_idx):
	videos = []
	dir = os.path.expanduser(dir)
	for target in sorted(os.listdir(dir)):
		d = os.path.join(dir,target)
		if not os.path.isdir(d):
			continue	
		for v_name in sorted(os.listdir(d)):
			video_path = os.path.join(d,v_name)
			if os.path.isdir(video_path):
				item = (video_path,class_to_idx[target])
				videos.append(item)

	return videos





def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root,extensions,loader=default_loader,seq_num=32,transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_video_list(root, class_to_idx)
        # print('----------samples----------')
        # print(samples)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.seq_num = seq_num

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        extensions = self.extensions
        path, target = self.samples[index]

        file_list = [x for x in os.listdir(path) if x.find('DS_Store')==-1] #获得该视频文件夹下的所有图片文件
        
        while len(file_list) < self.seq_num: #若该视频帧数小于目标帧数，则直接增广直至符合要求
        	file_list = file_list + file_list

        random_list = random.sample(file_list,self.seq_num) #从图片文件中随机抽取需要的seq_num张
        # print(random_list)

        First_pic = True
        for item in random_list:
        	pic_path = os.path.join(path,item)
        	pic = self.loader(pic_path)
        	if self.transform is not None:
        		pic = self.transform(pic)
        		pic = pic.view(1,3,112,96)
        		if First_pic:
        			pics = pic
        			First_pic = False
        		else:
        			pics = torch.cat([pics,pic],0)

        if self.target_transform is not None:
        	target = self.target_transform(target)

        return pics, target


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


