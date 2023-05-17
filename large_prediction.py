from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D

import skimage.io as io

import time

np.random.seed(6)
lbl_cmap = random_label_cmap()

S = sorted(glob('data/ctx_dapi/test/images/*quaterF*.tif'))
X = list(map(imread,S))

n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))


demo_model = False

if demo_model:
    print (
        "NOTE: This is loading a previously trained demo model!\n"
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist3D.from_pretrained('3D_demo')
else:
    model = StarDist3D(None, name='stardist', basedir='models')
None;


#root='./data/ctx_dapi/test/result/'
def example(model, i, show_dist=True):
    root=S[i].replace('images','r1')
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