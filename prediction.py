from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D
import os
import skimage.io as io

import time
import argparse
np.random.seed(6)
lbl_cmap = random_label_cmap()

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,default="*DAPI.tif")
parser.add_argument("-m","--model",type=str,default="stardist")
parser.add_argument("-t","--target",type=str,default="r1")
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, enable=True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#with tf.device('/gpu:0'):
S = sorted(glob('data/ctx_dapi/test/images/'+args.path))
X = list(map(imread,S))

n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))



model = StarDist3D(None, name=args.model, basedir='models')


#root='./data/ctx_dapi/test/result/'
def example(model, i, show_dist=True):
    root=S[i].replace('images',args.target)
    if not os.path.exists(root):
        os.makedirs(root)
    root=root.replace('.tif','_predict.tif')
    img = normalize(X[i], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)
    io.imsave(root,labels)
    root_point=root.replace('tif','txt')
    with open(root_point, 'w') as f:
        for line in details['points']:
            f.write(str(line[0])+','+str(line[1])+','+str(line[2])+'\n')
        
start=time.time()

for i in range(len(X)):
    print(i,'/',len(X))
    example(model,i)

end=time.time()
print('time:',str((end-start)/60),'s')