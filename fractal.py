import cv2
import numpy as np

# 手动修补 numpy 2.0 兼容性问题
import sys
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }

# 修补其他可能的兼容性问题
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int0'):
    np.int0 = np.intp

import imgaug.augmenters as iaa
from tqdm import trange

PATH='../datasets/ssv/trainS/'

def rectangle(x,y,w,h,img,flag=-1,ite=3):
    epsx=np.random.randint(-25,25)
    epsy=np.random.randint(-25,25)
    if x<0:
        x=0
    if y<0:
        y=0
    if x+w>511:
        x2=511
    else:
        x2=x+w
    if y+h>511:
        y2=511
    else:
        y2=y+h
    img[int(x):int(x2),int(y):int(y2)]=255
    flag*=-1
    ite-=1
    if ite>0:
      if flag==1:
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*h/6))
        img=rectangle(x-w2,y+h/3,w2,h2,img,flag,ite)
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*h/6))
        img=rectangle(x,y+eps,w2,h2,img,flag,ite)
      else:
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*w/6))
        img=rectangle(x+eps,y-h2,w2,h2,img,flag,ite)
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*w/6))
        img=rectangle(x+eps,y+h,w2,h2,img,flag,ite)
    return img


# 创建 imgaug 的增强序列 - 与原 IAAPiecewiseAffine 完全一致
piecewise_affine = iaa.PiecewiseAffine(
    scale=(0.09, 0.13), 
    nb_rows=4, 
    nb_cols=4, 
    order=1, 
    cval=0, 
    mode='constant'
)

rotate = iaa.Sometimes(0.5, iaa.Rotate((-30, 30)))

# 组合增强
aug_imgaug = iaa.Sequential([piecewise_affine, rotate])

for i in trange(0,1621):
    img=np.zeros((512,512), dtype=np.uint8)
    x=np.random.randint(10,250)
    w=np.random.randint(15,20)  # (15,25)
    y=np.random.randint(2,80)
    h=np.random.randint(350,450)
    ite=np.random.randint(3,5)
    img=rectangle(x,y,w,h,img,-1,ite)
    
    # 使用 imgaug 进行增强 - 功能与原 IAAPiecewiseAffine 完全一致
    img = aug_imgaug(image=img)
    
    ipath = PATH + str(i).zfill(5) + '.png'
    cv2.imwrite(str(ipath), img)
