clear;clc;
video_dir_path = '../videos';
video_size = [176 144];
dim_s = video_size(1) * video_size(2) * 3 / 2;
size_spa=32;
stride_spa=14;
stride_tem=8;
size_tem=20; % the size includes the frames used in optical flow.
size_seq = 10;
real_seq = size_tem -  size_seq + 1;
side_seq = (size_tem - size_seq) / 2;
scale = 4;
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
        y = single(fread(fid, video_size, 'uint8')) / 255;
        Y(j,:,:) = y;
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

hr_data = zeros([h_s * w_s * als, size_seq, 1, size_spa, size_spa]);
lr_data = zeros([h_s * w_s * als, size_seq, real_seq, size_spa, size_spa]);


cn = 1;
tmp = zeros(video_size);
cen = zeros(video_size);
para = get_para_flow(video_size(1),video_size(2));
kk = zeros([real_seq, video_size]);
for i = 1 : length(data)
    lr = data{i}.data;
    hr = data{i}.origin;
    seq = data{i}.seq;
    for q = 1 : seq
        ss = lr((q-1)*stride_tem+1:(q-1)*stride_tem+size_tem, :, :); % the len is size_tem
        an = 1;
        for p = side_seq+1 : size_tem -  side_seq % the len is real_seq
            cen(:,:) = ss(p, :, :);
            kk = ss(p - side_seq:p + side_seq, :, :);
            for l = 1 : real_seq
                if l ~= side_seq+1
                    tmp(:, :) = kk(l, :, :);
                    F = LDOF(tmp,cen,para,0);
                    for m = 1:video_size(1)
                        for n = 1:video_size(2)
                            kk(l, max(1, min(video_size(1), m + int64(F(m, n, 2)))), max(1, min(video_size(2), n + int64(F(m, n, 1))))) = tmp(m, n);
                        end
                    end
                end
            end
            for j = 1:h_s
                for k = 1:w_s
                    lr_c = kk(:, (j-1)*stride_spa+1:(j-1)*stride_spa+size_spa, (k-1)*stride_spa+1:(k-1)*stride_spa+size_spa);
                    hr_c = cen((j-1)*stride_spa+1:(j-1)*stride_spa+size_spa, (k-1)*stride_spa+1:(k-1)*stride_spa+size_spa);
                    lr_data(cn, an, :, :, :) = lr_c;
                    hr_data(cn, an, 1, :, :) = hr_c;
                end
            end
            an = an + 1;
        end
        cn = cn + 1;
    end
    fprintf('crop with the %d sequence\n', i);
end

save(strcat('../data/', num2str(length(filepaths)), '_seq_', ...
num2str(h_s * w_s * als), '_yuv_scala_', num2str(scale), '_frm', num2str(size_tem), '_blur_', num2str(blur_size),'_of.mat'), ...
'hr_data', 'lr_data', '-v7.3');

fprintf('...make done\n');
