function prepare_hygrip_trials(input_h5, output_dir, max_subjects)
% Prepare aligned HYGRIP EEG/HbO/HbR trials for three baseline models.

if nargin < 1 || isempty(input_h5)
    input_h5 = 'D:\DataSets\HYGRIP\hygrip.h5';
end
if nargin < 2 || isempty(output_dir)
    output_dir = 'D:\data\HYGRIP-Baselines\prepared';
end
if nargin < 3 || isempty(max_subjects)
    max_subjects = 14;
end
if ~isfile(input_h5)
    error('Missing input HDF5: %s', input_h5);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

eeg_fs = double(h5readatt(input_h5, '/', 'eeg_sfreq'));
oxy_fs = double(h5readatt(input_h5, '/', 'oxy_sfreq'));
oxy_unit = char(h5readatt(input_h5, '/', 'oxy_units'));
dxy_unit = char(h5readatt(input_h5, '/', 'dxy_units'));
if eeg_fs ~= 1000 || oxy_fs ~= 12.5
    error('Unexpected sampling rates EEG=%g, fNIRS=%g', eeg_fs, oxy_fs);
end
if ~strcmp(oxy_unit, 'mol') || ~strcmp(dxy_unit, 'mol')
    error('Expected author-provided Hb concentration in mol, got %s/%s', oxy_unit, dxy_unit);
end

subjects = 'ABCDEFGHIJKLMN';
max_subjects = min(max_subjects, numel(subjects));
[b_hb, a_hb] = butter(3, [0.01 0.1] / (oxy_fs / 2), 'bandpass');
manifest = struct([]);

for si = 1:max_subjects
    subject = subjects(si);
    prefix = ['/' subject];
    eeg = double(h5read(input_h5, [prefix '/eeg']));
    hbo = double(h5read(input_h5, [prefix '/oxy']));
    hbr = double(h5read(input_h5, [prefix '/dxy']));
    eeg_events = double(h5readatt(input_h5, [prefix '/eeg'], 'events'));
    hb_events = double(h5readatt(input_h5, [prefix '/oxy'], 'events'));

    eeg_keep = eeg_events(2, :) ~= -1;
    hb_keep = hb_events(2, :) ~= -1;
    eeg_labels = int64(eeg_events(2, eeg_keep));
    hb_labels = int64(hb_events(2, hb_keep));
    if ~isequal(eeg_labels, hb_labels)
        error('EEG/fNIRS label mismatch for subject %c', subject);
    end
    labels = eeg_labels(:);
    n_trials = numel(labels);
    if sum(labels == 0) ~= sum(labels == 1)
        error('Unbalanced labels for subject %c', subject);
    end

    % HbO/HbR are author-provided molar concentration signals. Filter the
    % continuous records before epoching, then baseline-correct each trial.
    hbo = filtfilt(b_hb, a_hb, hbo);
    hbr = filtfilt(b_hb, a_hb, hbr);
    eeg_uv = zeros(n_trials, 24, 4000, 'single');
    fnirs_um = zeros(n_trials, 2, 24, 250, 'single');
    eeg_times = eeg_events(1, eeg_keep);
    hb_times = hb_events(1, hb_keep);

    for ti = 1:n_trials
        eeg_start = round(eeg_times(ti) * eeg_fs) + 1;
        eeg_stop = eeg_start + 20 * eeg_fs - 1;
        if eeg_start < 1 || eeg_stop > size(eeg, 1)
            error('EEG epoch out of range: subject %c trial %d', subject, ti);
        end
        eeg_trial = resample(eeg(eeg_start:eeg_stop, :), 1, 5);
        if ~isequal(size(eeg_trial), [4000 24])
            error('Unexpected EEG output size for subject %c trial %d', subject, ti);
        end
        eeg_uv(ti, :, :) = single(eeg_trial' * 1000); % dataset mV -> uV

        hb_start = round(hb_times(ti) * oxy_fs) + 1;
        hb_stop = hb_start + 250 - 1;
        baseline_start = hb_start - round(oxy_fs);
        if baseline_start < 1 || hb_stop > size(hbo, 1)
            error('fNIRS epoch out of range: subject %c trial %d', subject, ti);
        end
        hbo_base = mean(hbo(baseline_start:hb_start-1, :), 1);
        hbr_base = mean(hbr(baseline_start:hb_start-1, :), 1);
        fnirs_um(ti, 1, :, :) = single((hbo(hb_start:hb_stop, :) - hbo_base)' * 1e6);
        fnirs_um(ti, 2, :, :) = single((hbr(hb_start:hb_stop, :) - hbr_base)' * 1e6);
    end

    meta = struct();
    meta.dataset = 'HYGRIP';
    meta.subject = subject;
    meta.task = 'dynamic grip force: left hand (0) vs right hand (1)';
    meta.epoch_seconds = [0 20];
    meta.eeg_original_sampling_rate_hz = eeg_fs;
    meta.eeg_output_sampling_rate_hz = 200;
    meta.eeg_input_unit = 'millivolts (dataset attribute spelling: milivolts)';
    meta.eeg_output_unit = 'microvolts';
    meta.fnirs_sampling_rate_hz = oxy_fs;
    meta.fnirs_input = 'author-provided oxy/dxy concentration';
    meta.fnirs_input_unit = 'mol';
    meta.fnirs_output_unit = 'umol/L';
    meta.fnirs_filter = '3rd-order Butterworth 0.01-0.1 Hz, zero-phase, continuous';
    meta.fnirs_baseline = '1 second immediately before task onset';
    meta.fnirs_motion_artifact_correction = 'none';
    meta.label_counts = [sum(labels == 0), sum(labels == 1)];

    output_file = fullfile(output_dir, sprintf('subject_%c_trials.mat', subject));
    save(output_file, 'eeg_uv', 'fnirs_um', 'labels', 'meta', '-v7');
    manifest(si).subject = subject; %#ok<AGROW>
    manifest(si).output_file = output_file;
    manifest(si).trials = n_trials;
    manifest(si).left = sum(labels == 0);
    manifest(si).right = sum(labels == 1);
    manifest(si).eeg_shape = size(eeg_uv);
    manifest(si).fnirs_shape = size(fnirs_um);
    fprintf('[%02d/%02d] subject %c: trials=%d EEG=%s fNIRS=%s\n', ...
        si, max_subjects, subject, n_trials, mat2str(size(eeg_uv)), mat2str(size(fnirs_um)));
    clear eeg hbo hbr eeg_uv fnirs_um;
end

save(fullfile(output_dir, 'manifest.mat'), 'manifest', '-v7');
fid = fopen(fullfile(output_dir, 'PREPROCESSING_ASSUMPTIONS.txt'), 'w');
fprintf(fid, ['HYGRIP author-provided oxy/dxy were used directly as HbO/HbR.\n' ...
    'No optical-density or MBLL reconstruction was applied.\n' ...
    'fNIRS: continuous 0.01-0.1 Hz Butterworth filtering, then 1 s pre-onset baseline.\n' ...
    'EEG: task-onset 0-20 s, resampled 1000->200 Hz, mV->uV.\n' ...
    'No ICA or motion-artifact correction was applied in this baseline.\n']);
fclose(fid);
fprintf('Prepared %d HYGRIP subjects in %s\n', max_subjects, output_dir);
end
