function clean_run_results(roots, alsoRemoveStale)
% CLEAN_RUN_RESULTS  Remove the accumulated campaign array from run results files.
%
% Eleven of the thirteen run results files were saved with the campaign loop's
% workspace still in scope, so each carries an `exps` cell array holding every
% earlier run's data. That is what makes ExpLCL_4_05_results.mat 840 MB when its
% own content is about 125 MB. The run's own data is unaffected: it lives in the
% top-level Surfs, PIV and USurf structs, and no file contains itself in its own
% `exps`, which is checked below before anything is removed.
%
% Deleting the variable in MATLAB (rather than unlinking it with h5py) rebuilds
% the file from the in-memory structure, so the orphaned 695 MB in /#refs# is
% simply never written and the MATLAB string objects in Surfs.paths are
% reconstituted correctly.
%
% The outgoing file is copied to a name carrying its own modification time,
% matching the convention already used in these directories.
%
%   clean_run_results(roots)              % remove `exps` only
%   clean_run_results(roots, true)        % also remove the stale loose variables
%
% `roots` is a cellstr of full paths to <exp>_results.mat files.
%
% ALSOREMOVESTALE additionally drops the loose top-level copies of the surface
% variables. These are NOT this run's data: in ExpLCL_4_05 the top-level
% surfsPIV is byte-identical to ExpLCL_4_03's, i.e. the previous loop
% iteration's. They shadow the correct values inside Surfs and are a trap for
% anyone reading the file. Off by default because removing them is a judgement
% call, not a cleanup.

    if nargin < 2, alsoRemoveStale = false; end

    for k = 1:numel(roots)
        file = roots{k};
        fprintf('\n=== %s\n', file);
        if ~isfile(file)
            fprintf('  missing, skipped\n'); continue
        end

        info = dir(file);
        before.bytes = info.bytes;
        stamp = datetime(info.date); stamp.Format = 'yyyy-MM-dd_HHmmss';
        [folder, base, ext] = fileparts(file);
        backup = fullfile(folder, [base '_' char(stamp) ext]);
        if isfile(backup)
            fprintf('  backup %s already exists; skipped to avoid overwriting\n', ...
                    [base '_' char(stamp) ext]);
            continue
        end

        S = load(file);
        if ~isfield(S, 'exps') && ~alsoRemoveStale
            fprintf('  no `exps` field; nothing to do\n'); continue
        end

        % Refuse to touch a file that stores its own run inside `exps`, which
        % would make the array load-bearing rather than redundant.
        names = {};
        if isfield(S, 'exps')
        for jj = 1:numel(S.exps)
            e = S.exps{jj};
            if isstruct(e) && isfield(e, 'exp_name'), names{end+1} = e.exp_name; end %#ok<AGROW>
        end
        end
        if isfield(S, 'exp_name') && any(strcmp(names, S.exp_name))
            fprintf('  REFUSING: %s appears inside its own `exps`\n', S.exp_name);
            continue
        end
        if ~isempty(names)
            fprintf('  exps holds %d runs: %s\n', numel(names), strjoin(names, ', '));
        end

        % Fingerprint what must survive, so the saved file can be checked.
        keep = {'Surfs','PIV','USurf','exp_name','number_of_pair'};
        fp = struct();
        for f = keep
            if isfield(S, f{1}), fp.(f{1}) = fingerprint(S.(f{1})); end
        end

        copyfile(file, backup);
        b = dir(backup);
        if b.bytes ~= before.bytes
            error('Backup size differs from the original; aborting before any write.');
        end
        fprintf('  backed up to %s (%.0f MB)\n', [base '_' char(stamp) ext], b.bytes/1e6);

        removed = {};
        if isfield(S, 'exps'), S = rmfield(S, 'exps'); removed = {'exps'}; end
        if alsoRemoveStale
            % Remove a loose top-level variable only where it can be PROVED
            % wrong: it must also exist inside Surfs and disagree with it.
            % Anything that matches is a harmless duplicate and is left alone,
            % and anything with no counterpart to check against is left alone
            % too, because there would be no evidence either way.
            drop = {}; kept = {};
            for f = fieldnames(S)'
                name = f{1};
                if strcmp(name, 'Surfs') || ~isfield(S.Surfs, name), continue, end
                if isequaln(S.(name), S.Surfs.(name))
                    kept{end+1} = name; %#ok<AGROW>
                else
                    drop{end+1} = name; %#ok<AGROW>
                end
            end
            if ~isempty(drop)
                S = rmfield(S, drop);
                removed = [removed, drop]; %#ok<AGROW>
            end
            if ~isempty(kept)
                fprintf('  left alone (identical to Surfs, so harmless): %s\n', strjoin(kept, ', '));
            end
        end

        save(file, '-struct', 'S', '-v7.3');

        % Verify: everything that had to survive must be bit-identical.
        T = load(file);
        bad = {};
        for f = fieldnames(fp)'
            if ~isfield(T, f{1}) || ~isequal(fingerprint(T.(f{1})), fp.(f{1}))
                bad{end+1} = f{1}; %#ok<AGROW>
            end
        end
        if ~isempty(bad)
            error('RETAINED DATA CHANGED: %s -- restore from %s', strjoin(bad, ', '), backup);
        end
        after = dir(file);
        fprintf('  removed: %s\n', strjoin(removed, ', '));
        fprintf('  verified %d retained structs unchanged\n', numel(fieldnames(fp)));
        fprintf('  %.0f MB -> %.0f MB (%.1fx smaller)\n', ...
                before.bytes/1e6, after.bytes/1e6, before.bytes/after.bytes);
    end
    fprintf('\nCLEAN_RUN_RESULTS_DONE\n');
end


function h = fingerprint(value)
% A cheap structural+numeric signature, recursing into structs.
    if isstruct(value)
        % A struct ARRAY must be walked element by element: value.(field) on one
        % returns a comma-separated list, not a value.
        h = sprintf('struct%s', mat2str(size(value)));
        for ii = 1:numel(value)
            element = value(ii);
            for f = sort(fieldnames(element))'
                h = [h '|' f{1} ':' fingerprint(element.(f{1}))]; %#ok<AGROW>
            end
            if ii >= 4
                % Long directory listings add nothing; their count is enough.
                h = [h sprintf('|...%d more', numel(value)-ii)]; break %#ok<AGROW>
            end
        end
    elseif isnumeric(value) || islogical(value)
        v = double(value(:));
        v = v(isfinite(v));
        h = sprintf('%s n=%d sum=%.12g', mat2str(size(value)), numel(v), sum(v));
    else
        h = sprintf('%s %s', class(value), mat2str(size(value)));
    end
end
