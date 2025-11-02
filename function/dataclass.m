function [tstNum,num,part,trnData,trnLab,tstData,tstLab ] = dataclass(data,label,N_class )
    trnPer = 0.1; 
    [num,part,trnData,trnLab,tstData,tstLab] = TrainTest(data,label,trnPer,N_class);
    tstNum = zeros(1,N_class);
    for i = 1:N_class
        index = find(tstLab == i);
        tstNum(i) = length(index);
    end
end

