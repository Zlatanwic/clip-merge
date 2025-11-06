addpath(genpath('steerable_pyramid-master'));

img1=double(imread('g_06_bf.tif'));
img2=double(imread('g_06_dsift.tif'));

vif = vifvec(img1, img2);