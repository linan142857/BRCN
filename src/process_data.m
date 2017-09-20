clear;clc;
video_dir_path = '../videos';
video_size = [176 144];
dim_s = video_size(1) * video_size(2) * 3 / 2;
size_spa=44;
stride_spa=14;
stride_tem=8;
size_tem=22;
scale = 2;
blur_size = 2;

filepaths = dir(fullfile(video_dir_path,'*.yuv'));

if ~exist('../data')
        mkdir('../data');
end

alf = 0;
als = 0;
h_s = fix((video_size(1) - size_spa) / stride_spa) + 1;
w_s = fix((video_size(2) - size_spa) / stride_spa) + 1;
H = fspecial('gaussian', [5 5], blur_size);

% H = lpfilter('gaussian', video_size(1), video_size(2), blur_size);

data = cell(1, length(filepaths));
fprintf('... %d video sequences\n', length(filepaths));
for i = 1:length(filepaths)
    filepath = fullfile(video_dir_path, filepaths(i).name);
    file = dir(filepath);
    numfrm = file.bytes / dim_s;
    seq = fix((numfrm - size_tem) / stride_tem) + 1;
    alf = alf + numfrm;
    als = als + seq;
    fprintf('blurring... the %d video sequences, ... %d frms\n', i, numfrm);
    fid = fopen(filepath, 'r');
    Y = zeros([numfrm, video_size]);
    X = zeros([numfrm, video_size]);
    for j = 1:numfrm
        y = fread(fid, video_size, 'uint8') ./ 255;
        Y(j,:,:) = y;
%         F = fft2(y);
%         G = H.*F;
%         g = abs(ifft2(G));
        g = imfilter(y, H, 'corr', 'replicate');
        g = imresize(imresize(g, 1/scale, 'bicubic'), scale, 'bicubic');
        X(j,:,:) = g;
        fread(fid, [video_size(1), video_size(2) / 2], 'uint8');
    end
    fclose(fid);

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

an = 1;
cn = 1;
for i = 1:length(data)
    lr = data{i}.data;
    hr = data{i}.origin;
    seq = data{i}.seq;
    for q = 1:seq

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

save(strcat('../data/', num2str(length(filepaths)), '_seq_', ...
num2str(h_s * w_s * als), '_yuv_scala_', num2str(scale), '_frm', num2str(size_tem), '_blur_', num2str(blur_size),'.mat'), ...
'hr_data', 'lr_data', '-v7.3');

fprintf('...make done\n');
