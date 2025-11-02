function fit = fitness(net,position,N_bands,final_groups,IE)
bands =  decoding(position,final_groups);
mask = zeros(1,N_bands);
mask(1,bands)=1;
mask = mask + IE;
fit =1- net(mask');
end