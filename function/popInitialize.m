function X = popInitialize(N_pop,N_bands,N_sel,S_band)
    m = fix(N_bands/N_sel);
    n = mod(N_bands,N_sel);
    X = zeros(N_pop,N_bands);
    for j = 1:N_pop
        pos=[];
        for i = 1:N_sel
            Sc =rand(m,1) .* S_band(m*(i-1)+1:m*i);
            Sc = (Sc - min(Sc)) ./ (max(Sc) - min(Sc));
            pos = [pos,Sc'];
        end
        Sc=S_band(N_sel*m+1:N_sel*m+n);
        Sc = (Sc - min(Sc)) ./ (max(Sc) - min(Sc));
        pos = [pos,Sc'];
        X(j,:)=pos;
    end
end