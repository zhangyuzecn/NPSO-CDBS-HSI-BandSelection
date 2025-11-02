function gbest = findgbest(pop)
    [F,R] = fastNonDominatedSort(pop);
    F1 = F{1};
    [~,min_idx] = min(pop(F1,1));
    gbest = F1(min_idx);
end