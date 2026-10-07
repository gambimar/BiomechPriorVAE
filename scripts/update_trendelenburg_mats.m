function update_trendelenburg_mats()
    addpath(genpath('/home/rzlin/ri94mihu/phd/BiomechPriorVAE/'));
    
    levels = [25, 50, 75, 90];

    for idx = 1:length(levels)
        pct = levels(idx);
        
        osimFile = sprintf('data/model/sipp_generic_runmad_smoothsphere_trendelenburg%d.osim', pct);
        maFile   = sprintf('data/model/sipp_generic_runmad_smoothsphere_trendelenburg%d_momentarms.mat', pct);

        % Compute exact SHA-256 hash byte-array as done in Gait3d.m
        fid = fopen(osimFile);
        fileID_read = fread(fid);
        fclose(fid);
        md = java.security.MessageDigest.getInstance('SHA-256');
        osim_sha256_new = typecast(md.digest(uint8(fileID_read))', 'uint8');

        % Update moment arms mat file sha256
        d_ma = load(maFile);
        d_ma.osim_sha256 = osim_sha256_new;
        save(maFile, '-struct', 'd_ma');

        fprintf('Successfully updated %s (sha256 matched)\n', maFile);
    end
end
