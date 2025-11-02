function S_band = calBandScore(N_bands,net)
    S_band=zeros(N_bands,1);
    for i = 1:N_bands
        mask = zeros(N_bands,1);
        mask(i,1)=1;
        S_band(i)=net(mask);
    end
end