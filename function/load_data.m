
%% load data

function [data,label,IE] = load_data(data_type)

if strcmp(data_type, 'PA')
    load data/QUH-Pingan.mat
    load data/QUH-Pingan_GT.mat
    data = Haigang;
    label = HaigangGT;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
end

if strcmp(data_type, 'QY')
    load data/QUH-Qingyun.mat
    load data/QUH-Qingyun_GT.mat
    data = Chengqu;
    label = ChengquGT;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
end

if strcmp(data_type, 'TD')
    load data/QUH-Tangdaowan.mat
    load data/QUH-Tangdaowan_GT.mat
    data = Tangdaowan;
    label = TangdaowanGT;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
end




[M,N]=size(label);
label = reshape(label,M*N,1);

label_All = label(:);
label_inds = find(label_All > 0);
label = label(label_inds);

sz = size(data);
data = reshape(data,sz(1)*sz(2),sz(3));
data = double(data(label_inds, :));

data = mapminmax(data,0,1);
data = data +0.0001;



