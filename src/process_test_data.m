clear;clc;
video_dir_path = '../videos/test';
video_size = [241 311];
size_spa=32;
stride_spa=14;
stride_tem=8;
size_tem=10;
scale = 4;
name = 'Star_Fan'

filepaths = dir(fullfile(video_dir_path,'Star_Fan.mp4'));

if ~exist('../data/test')
        mkdir('../data/test');
end

alf = 0;
als = 0;
k = 1;
h_s = fix((video_size(1) - size_spa) / stride_spa) + 1;
w_s = fix((video_size(2) - size_spa) / stride_spa) + 1;
H = lpfilter('gaussian', video_size(1), video_size(2), 30);

data = cell(1, length(filepaths));
fprintf('... %d video sequences\n', length(filepaths));
for i = 1:length(filepaths)
    filepath = fullfile(video_dir_path, filepaths(i).name);
    fid = VideoReader(filepath);
    numfrm = 300;
    nFrames = fid.NumberOfFrames;
    seq = fix((numfrm - size_tem) / stride_tem) + 1;
    alf = alf + numfrm;
    als = als + seq;
    fprintf('blurring... the %d video sequences, ... %d frms\n', i, numfrm);
    Y = zeros([numfrm, video_size]);
    X = zeros([numfrm, video_size]);
    for j = 1:numfrm
        y = read(fid, j);
        y = rgb2ycbcr(y);
        y = double(y(200:440, 340:650, 1)) ./ 255;
        Y(k,:,:) = y;
        
%         F = fft2(y);
%         G = H.*F;
%         g = abs(ifft2(G));
%         g = imresize(imresize(g, 1/scale), video_size, 'bicubic');
        f = imresize(imresize(y, 1/scale), video_size, 'bicubic');
        X(k,:,:) = f;
        k = k + 1;
    end

    data{i} = struct();
    data{i}.seq = fix((numfrm - size_tem) / stride_tem) + 1;
    data{i}.name = filepaths(i).name;
    data{i}.data = X;
    data{i}.origin = Y;
end
fprintf('...loading done, %d frms\n\nbeginning cropping frame\n', alf);
fprintf('...all cropped number: %d\n', h_s * w_s * als);

hr_data = zeros([h_s * w_s * als, size_tem, 1, size_spa, size_spa]);
lr_data = zeros([h_s * w_s * als, size_tem, 1, size_spa, size_spa]);
X = zeros([als, size_tem, 1, video_size]);
Y = zeros([als, size_tem, 1, video_size]);
an = 1;
cn = 1;
for i = 1:length(data)
    lr = data{i}.data;
    hr = data{i}.origin;
    seq = data{i}.seq;
    for q = 1:seq
        X(an, :, 1, :, :) = lr((q-1)*stride_tem+1:(q-1)*stride_tem+size_tem, :, :);
        Y(an, :, 1, :, :) = hr((q-1)*stride_tem+1:(q-1)*stride_tem+size_tem, :, :);
        an = an + 1;
        for j = 1:h_s
            for k = 1:w_s
               lr_c = lr((q-1)*stride_tem+1:(q-1)*stride_tem+size_tem,...
                   (j-1)*stride_spa+1:(j-1)*stride_spa+size_spa,...
                   (k-1)*stride_spa+1:(k-1)*stride_spa+size_spa);
               hr_c = hr((q-1)*stride_tem+1:(q-1)*stride_tem+size_tem,...
                   (j-1)*stride_spa+1:(j-1)*stride_spa+size_spa,...
                   (k-1)*stride_spa+1:(k-1)*stride_spa+size_spa);
               hr_data(cn, :, 1, :, :) = hr_c;
               lr_data(cn, :, 1, :, :) = lr_c;
               cn = cn + 1;
            end
        end
    end
    fprintf('crop with the %d sequence\n', i);
end

save(strcat('../data/test/', num2str(h_s * w_s * als), '_yuv_scala_', num2str(scale), '_frm', num2str(size_tem), '_', name,'.mat'), ...
'hr_data', 'lr_data', '-v7.3');
fprintf('...make done\n');