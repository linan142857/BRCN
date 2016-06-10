x = Y(1:6, :, :);
op = zeros(15, 176, 144, 2);
k = 1
for i = 1:5
    for j = i+1:6
        [op(k, :, :, :),~,~]=LDOF(squeeze(x(i, :, :)), squeeze(x(j, :, :)), para,0);
        k = k +1
    end
end
opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
