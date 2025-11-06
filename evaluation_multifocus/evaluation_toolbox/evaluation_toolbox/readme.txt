1. metricYang.m:  A novel similarity based quality metric for image fusion, Information Fusion, Vol.9, pp156-160, 2008, by Cui Yang et al.


2. metricCvejic.m: paremeter 'sw' set to 2. A Similarity Metric for Assessment of Image Fusion Algorithms, International Journal of Information and Communication Engineering 2 (3) 2006, pp.178-182. by N. Cvejic et al.


3. metricZhao.m, myphasecong3.m, lowpassfilter.m:  Performance assessment of combinative pixel-level image fusion based on an absolute feature measurement, International Journal of Innovative Computing, Information and Control, 3 (6A) 2007, pp.1433-1447. by J. Zhao et al. 


4. metricMI.M,normalize1.m, mutual_info.m,tsallis.m : default 'sw=1' Comments on "Information measure for performance of image fusion". By M. Hossny et al Electronics Letters Vol. 44, No.18, 2008 'sw=3' Image fusion metric based on mutual information and Tsallis entropy. By N. Cvejie et al. Electronics Letters, Vol.42, No. 11, 2006

5. metricChenBlum.m: A new automated quality assessment algorithm for image fusion, Image and Vision Computing, 27 (2009) 1421-1432. By Yin Chen et al.


NCC.m and ssim_index.m are not used.



%-----------------------------------------------------------------------

Contents:

[1] - metricMI.m: revised mutual information based metric; Tsallis entropy (Cvejic & Nava);  
[2] - metricWang.m: nonlinear correlation information entropy;
[3] - metricXydeas.m: Xydeas's fusion metric;
[4] - metricPWW.m: multi-scale fusion metric; 
(*Simoncelli's steerable pyramid toolbox is used, which is available at: http://www.cns.nyu.edu/~eero/steerpyr/. However, you can use the functions from Matlab wavelet toolbox. Need a little bit modification of this function.)
[5] - metricZheng.m: metric based on spatial frequency;
[6] - metricZhao.m: image phase congruency based metric;
(This function uses the phase congruency implementation [phasecong2.m] from Dr. Peter Kovesi, available at http://www.csse.uwa.edu.au/~pk/research/matlabfns/. The modified version myphasecong3.m is used in this fusion metric function.)
[7] - metricPeilla.m: Peilla's metric;
(This metric uses the ssim_index function by Dr. Zhou Wang, available at https://ece.uwaterloo.ca/~z70wang/research/ssim/.)
[8] - metricCvejic.m: two fusion metric by Cvejic;
[9] - metricYang.m: image similarity based metric;
[10] - metricChen.m: human perception inspired qualtiy metric (by Dr. Hao Chen);
[11] - metricChenBlum.m: fusion metric by Dr. Yin Chen. 
