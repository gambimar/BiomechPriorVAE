function run_range_winter_bhargava(start_idx, end_idx)
    addpath(genpath('/home/rzlin/ri94mihu/phd/BiomechPriorVAE/'));
    addpath(genpath('/home/rzlin/ri94mihu/phd/mexIPOPT'));
    addpath('scripts');
    
    for i = start_idx:end_idx
        fprintf('\n========================================\n');
        fprintf('Worker executing run %02d / %02d\n', i, end_idx);
        fprintf('========================================\n');
        try
            run_single_winter_bhargava(i);
        catch ME
            fprintf('Error in run %02d: %s\n', i, ME.message);
        end
    end
end
