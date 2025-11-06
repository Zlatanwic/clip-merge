function out= MS_SSIM(image_ir,image_vis,image_f)
[s1,s2] = size(image_ir);
imgSeq = zeros(s1, s2, 2);
imgSeq(:, :, 1) = image_ir;
imgSeq(:, :, 2) = image_vis;

[out,t1,t2]= analysis_ms_ssim(imgSeq, image_f);

end