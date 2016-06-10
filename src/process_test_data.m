clear;clc;
video_dir_path = '../videos/test';
video_size = [241 311;181 201;161 174;168 211;308 475];
scale = 4;
blur_size = 2;
name = {'Star_Fan.mp4', 'Flag.mp4','Treadmill.mp4','Turbine.mp4','Dirty_Dancing.mp4'};
numfrms = [1 300; 1 290;1 300;1 350;49 106];
sc = [200 440 340 650;150 330 400 600;220 380 263 436;140 307 220 430;150 457 244 718];

if ~exist('../data/test')
        mkdir('../data/test');
end

H = fspecial('gaussian', [5 5], blur_size);

for i = 1:length(name)
    filepath = fullfile(video_dir_path, name{i});
    fid = VideoReader(filepath);
    numfrm = [numfrms(i, 1), numfrms(i, 2)];
    nf = numfrm(2) - numfrm(1) + 1;
    fprintf('blurring... the %d video sequences, ... %d frms\n', i, nf);
    v_s = [video_size(i,1), video_size(i,2)];
    v_s = floor(v_s / scale) * scale;
    region = [sc(i,1),sc(i,2),sc(i,3),sc(i,4)];
    k = 1;
    hr_data = zeros([1,nf, 1, v_s]);
    lr_data = zeros([1,nf, 1, v_s]);
    for j = numfrm(1):numfrm(2)
        y = read(fid, j);
        y = rgb2ycbcr(y);
        y = y(region(1):region(2), region(3):region(4), 1);
        y = double(y) / 255;
        hr_data(1,k,1,:,:) = modcrop(y, scale);
        g = imfilter(y, H, 'corr', 'replicate');
        g = modcrop(g, scale);
        g = imresize(imresize(g, 1/scale, 'bicubic'), scale, 'bicubic');
        lr_data(1,k,1,:,:) = g;
        k = k + 1;
    end
save(strcat('../data/test/', num2str(nf), '_', name{i},'_scale',num2str(scale),'_blur_2.mat'), ...
'hr_data', 'lr_data', '-v7.3');
end
fprintf('...make done\n');