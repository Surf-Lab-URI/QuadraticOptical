function add_qo_to_campaign(campaignFile, runRoot)
% ADD_QO_TO_CAMPAIGN  Copy the QO profile fields from each run file into exps.
%
% The campaign file holds the same per-run data as the individual run results
% files, organised as a cell array `exps`. The QO fields are written into the
% run files first (see write_qo_fields.py); this carries them across so the two
% stay consistent, taking each run's values from its own file rather than
% recomputing anything.
%
% Done in MATLAB rather than h5py for two reasons. Appending datasets to the
% existing HDF5 would grow the file by roughly 260 MB without reclaiming
% anything, and the per-run Surfs.paths are MATLAB string objects backed by
% /#subsystem#, which only MATLAB reconstitutes correctly.
%
% The outgoing file is copied to a name carrying its own modification time,
% matching the convention used in these directories.
%
%   add_qo_to_campaign('/.../Results_Surflab/results.mat', '/.../PIVdt10ms_IRlas1_8hz')

    FIELDS = {'U_QO','U_QOIR','z_QOIR','U_QOIRPIV','QO_commit','QO_acceptance', ...
              'QO_window_mm','QO_pairs_written','QO_written','QO_note'};

    assert(isfile(campaignFile), 'No campaign file at %s', campaignFile);
    info = dir(campaignFile);
    stamp = datetime(info.date); stamp.Format = 'yyyy-MM-dd_HHmmss';
    [folder, base, ext] = fileparts(campaignFile);
    backup = fullfile(folder, [base '_' char(stamp) ext]);
    assert(~isfile(backup), 'Backup %s already exists; refusing to overwrite', backup);

    fprintf('loading campaign file (%.0f MB)...\n', info.bytes/1e6);
    S = load(campaignFile);
    assert(isfield(S, 'exps'), 'Campaign file has no `exps`');

    % Fingerprint every run's existing content, so the save can be checked.
    before = cell(numel(S.exps), 1);
    for ii = 1:numel(S.exps)
        before{ii} = fingerprint_struct(S.exps{ii});
    end

    done = {}; skipped = {};
    for ii = 1:numel(S.exps)
        e = S.exps{ii};
        if ~isstruct(e) || ~isfield(e, 'exp_name'), skipped{end+1} = sprintf('entry %d', ii); continue, end %#ok<AGROW>
        name = e.exp_name;
        runFile = fullfile(runRoot, name, 'Results_Surflab', [name '_results.mat']);
        if ~isfile(runFile)
            fprintf('  %-14s no run file, skipped\n', name);
            skipped{end+1} = name; continue %#ok<AGROW>
        end
        R = load(runFile, 'PIV', 'exp_name');
        % The run file must be the run it claims to be before anything is copied.
        if isfield(R, 'exp_name') && ~strcmp(R.exp_name, name)
            error('%s contains exp_name %s; refusing to copy', runFile, R.exp_name);
        end
        missing = FIELDS(~isfield(R.PIV, FIELDS));
        if ~isempty(missing)
            fprintf('  %-14s run file has no QO fields yet (%s), skipped\n', name, strjoin(missing, ','));
            skipped{end+1} = name; continue %#ok<AGROW>
        end
        % The pair axis must line up, or the profiles would land on wrong times.
        assert(isequal(size(R.PIV.pairNum), size(e.PIV.pairNum)) && ...
               isequaln(R.PIV.pairNum, e.PIV.pairNum), ...
               '%s pairNum differs from the campaign entry', name);
        for f = FIELDS
            S.exps{ii}.PIV.(f{1}) = R.PIV.(f{1});
        end
        written = numel(R.PIV.QO_pairs_written);
        fprintf('  %-14s copied %d fields, %d pairs with data\n', name, numel(FIELDS), written);
        done{end+1} = name; %#ok<AGROW>
    end

    if isempty(done)
        fprintf('\nNothing to write; campaign file left untouched.\n');
        fprintf('\nADD_QO_TO_CAMPAIGN_DONE\n');
        return
    end

    copyfile(campaignFile, backup);
    b = dir(backup);
    assert(b.bytes == info.bytes, 'Backup size differs; aborting before any write.');
    fprintf('backed up to %s (%.0f MB)\n', [base '_' char(stamp) ext], b.bytes/1e6);

    save(campaignFile, '-struct', 'S', '-v7.3');

    % Verify: every run's pre-existing content must be unchanged, and the QO
    % fields must now be present in exactly the entries we wrote.
    T = load(campaignFile);
    bad = {};
    for ii = 1:numel(T.exps)
        after = fingerprint_struct(rmfield_if(T.exps{ii}, FIELDS));
        if ~isequal(after, before{ii}), bad{end+1} = sprintf('entry %d', ii); end %#ok<AGROW>
    end
    if ~isempty(bad)
        error('EXISTING CAMPAIGN DATA CHANGED: %s -- restore from %s', strjoin(bad, ', '), backup);
    end
    present = 0;
    for ii = 1:numel(T.exps)
        if isstruct(T.exps{ii}) && isfield(T.exps{ii}, 'PIV') && isfield(T.exps{ii}.PIV, 'U_QOIR')
            present = present + 1;
        end
    end
    a = dir(campaignFile);
    fprintf('wrote %d runs, skipped %d\n', numel(done), numel(skipped));
    fprintf('verified existing content unchanged in all %d entries\n', numel(T.exps));
    fprintf('U_QOIR now present in %d of %d entries\n', present, numel(T.exps));
    fprintf('%.0f MB -> %.0f MB\n', info.bytes/1e6, a.bytes/1e6);
    fprintf('\nADD_QO_TO_CAMPAIGN_DONE\n');
end


function s = rmfield_if(s, fields)
    if ~isstruct(s) || ~isfield(s, 'PIV'), return, end
    present = fields(isfield(s.PIV, fields));
    if ~isempty(present), s.PIV = rmfield(s.PIV, present); end
end


function h = fingerprint_struct(value)
    if isstruct(value)
        h = sprintf('struct%s', mat2str(size(value)));
        for ii = 1:numel(value)
            element = value(ii);
            for f = sort(fieldnames(element))'
                h = [h '|' f{1} ':' fingerprint_struct(element.(f{1}))]; %#ok<AGROW>
            end
            if ii >= 4
                h = [h sprintf('|...%d more', numel(value)-ii)]; break %#ok<AGROW>
            end
        end
    elseif isnumeric(value) || islogical(value)
        v = double(value(:)); v = v(isfinite(v));
        h = sprintf('%s n=%d sum=%.12g', mat2str(size(value)), numel(v), sum(v));
    else
        h = sprintf('%s %s', class(value), mat2str(size(value)));
    end
end
