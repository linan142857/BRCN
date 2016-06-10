clear;clc;
video_dir_path = '../videos';
img_dir_path = '../data/png';
video_size = [176 144];
dim_s = video_size(1) * video_size(2) * 3 / 2;
scale = 4;
if ~exist(img_dir_path)
        mkdir(img_dir_path);
end

im1_list =  fopen(strcat(img_dir_path, '/im1_list.txt'), 'w+');
im2_list =  fopen(strcat(img_dir_path, '/im2_list.txt'), 'w+');
flo_list  =  fopen(strcat(img_dir_path, '/flo_list.txt'), 'w+');

filepaths = dir(fullfile(video_dir_path,'*.yuv'));
fprintf('... %d video sequences\n', length(filepaths));
for i = 1:length(filepaths)
    filepath = fullfile(video_dir_path, filepaths(i).name);
    file = dir(filepath);
    numfrm = file.bytes / dim_s;
   fid = fopen(filepath, 'r');
    for j = 1:numfrm
       y = fread(fid, video_size, 'uint8') ./ 255;
       y = imresize(imresize(y, 1/scale, 'bicubic'), video_size, 'bicubic');
        fread(fid, [video_size(1), video_size(2) / 2], 'uint8');
        name = strcat(img_dir_path, '/', num2str(i), '_', num2str(j), '.png');
        name_1 = strcat(num2str(i), '_', num2str(j), '.png');
        name_2 = strcat(num2str(i), '_', num2str(j+1), '.png');
        name_flo = strcat(num2str(i), '_', num2str(j), '.flo');
        imwrite(y, name, 'png');
        if j ~= numfrm
            fprintf(im1_list, '%s\n', name_1);
            fprintf(im2_list, '%s\n', name_2);
            fprintf(flo_list, '%s\n', name_flo);
        end
    end
 fclose(fid);
end
fclose(im1_list);
fclose(im2_list);
fclose(flo_list);

fprintf('...make done\n');