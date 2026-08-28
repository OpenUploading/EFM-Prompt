function prepare_hygrip_trials_v2(input_h5, old_prepared_dir, output_dir, max_subjects)
% Corrected HYGRIP EEG preprocessing while preserving the verified fNIRS trials.
% EEG is processed continuously: downsample, robust despike, CAR, device-harmonic
% notches, and zero-phase 1-45 Hz bandpass. Existing outputs are never overwritten.

if nargin < 1 || isempty(input_h5), input_h5 = 'D:\DataSets\HYGRIP\hygrip.h5'; end
if nargin < 2 || isempty(old_prepared_dir), old_prepared_dir = 'D:\data\HYGRIP-Baselines\prepared'; end
if nargin < 3 || isempty(output_dir), output_dir = 'D:\data\HYGRIP-Baselines\prepared_eeg_v2'; end
if nargin < 4 || isempty(max_subjects), max_subjects = 14; end
if ~isfile(input_h5), error('Missing input HDF5: %s', input_h5); end
if ~isfolder(old_prepared_dir), error('Missing verified prepared data: %s', old_prepared_dir); end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

raw_fs = double(h5readatt(input_h5, '/', 'eeg_sfreq'));
if raw_fs ~= 1000, error('Expected EEG at 1000 Hz, got %g', raw_fs); end
target_fs = 200;
subjects = 'ABCDEFGHIJKLMN';
max_subjects = min(max_subjects, numel(subjects));
[sos_bp, g_bp] = butter(4, [1 45] / (target_fs / 2), 'bandpass');
harmonics = [12.5 25 37.5];
manifest = struct([]);

for si = 1:max_subjects
    subject = subjects(si);
    prefix = ['/' subject];
    raw = double(h5read(input_h5, [prefix '/eeg']));
    events = double(h5readatt(input_h5, [prefix '/eeg'], 'events'));
    keep = events(2, :) ~= -1;
    event_times = events(1, keep);
    event_labels = int64(events(2, keep));

    % Anti-aliased continuous downsampling and physical-unit conversion.
    % The HDF root attribute says "milivolts", but the authors' plot_eeg()
    % multiplies the stored EEG by 1e3 to display mV. The raw offset (~0.01)
    % and temporal variation (~1e-5) also identify the stored values as volts.
    eeg_uv_cont = resample(raw, 1, round(raw_fs / target_fs)) * 1e6;
    clear raw;

    % Robust continuous despiking per channel. At 12 robust SD this only
    % affects extreme acquisition artefacts, not ordinary EEG excursions.
    clipped_samples = 0;
    for ch = 1:size(eeg_uv_cont, 2)
        med = median(eeg_uv_cont(:, ch));
        robust_sd = 1.4826 * median(abs(eeg_uv_cont(:, ch) - med));
        robust_sd = max(robust_sd, eps);
        low = med - 12 * robust_sd;
        high = med + 12 * robust_sd;
        mask = eeg_uv_cont(:, ch) < low | eeg_uv_cont(:, ch) > high;
        clipped_samples = clipped_samples + sum(mask);
        eeg_uv_cont(:, ch) = min(max(eeg_uv_cont(:, ch), low), high);
    end

    % Common-average reference before zero-phase filtering.
    eeg_uv_cont = eeg_uv_cont - mean(eeg_uv_cont, 2);
    for f0 = harmonics
        w0 = f0 / (target_fs / 2);
        bw = w0 / 50;
        [b_notch, a_notch] = iirnotch(w0, bw);
        eeg_uv_cont = filtfilt(b_notch, a_notch, eeg_uv_cont);
    end
    eeg_uv_cont = filtfilt(sos_bp, g_bp, eeg_uv_cont);

    old_file = fullfile(old_prepared_dir, sprintf('subject_%c_trials.mat', subject));
    old = load(old_file, 'fnirs_um', 'labels', 'meta');
    labels = int64(old.labels(:));
    if ~isequal(labels, event_labels(:)), error('Label mismatch for subject %c', subject); end
    n_trials = numel(labels);
    eeg_uv = zeros(n_trials, 24, 20 * target_fs, 'single');
    for ti = 1:n_trials
        start_index = round(event_times(ti) * target_fs) + 1;
        stop_index = start_index + 20 * target_fs - 1;
        if start_index < 1 || stop_index > size(eeg_uv_cont, 1)
            error('EEG epoch out of bounds: subject %c trial %d', subject, ti);
        end
        eeg_uv(ti, :, :) = single(eeg_uv_cont(start_index:stop_index, :)');
    end
    fnirs_um = old.fnirs_um;
    meta = old.meta;
    meta.eeg_processing_version = 'v2_continuous_zero_phase';
    meta.eeg_processing = ['1000->200 Hz continuous resample; robust 12-MAD clipping; ' ...
        'CAR; 12.5/25/37.5 Hz notches Q=50; zero-phase Butterworth order-4 1-45 Hz; epoch 0-20 s'];
    meta.eeg_source_unit_note = ['HDF attribute says milivolts, but author plot_eeg multiplies raw by 1e3 to plot mV; ' ...
        'raw magnitude also supports volts. Converted V->uV by 1e6.'];
    meta.eeg_clipped_fraction = clipped_samples / numel(eeg_uv_cont);
    meta.eeg_normalization = 'none in stored physical uV; model-specific normalization at load time';
    output_file = fullfile(output_dir, sprintf('subject_%c_trials.mat', subject));
    save(output_file, 'eeg_uv', 'fnirs_um', 'labels', 'meta', '-v7');

    manifest(si).subject = subject; %#ok<AGROW>
    manifest(si).output_file = output_file;
    manifest(si).trials = n_trials;
    manifest(si).clipped_fraction = meta.eeg_clipped_fraction;
    manifest(si).eeg_std_uv = std(double(eeg_uv(:)));
    fprintf('[%02d/%02d] subject %c: trials=%d clipped=%.6f std=%.4f uV\n', ...
        si, max_subjects, subject, n_trials, manifest(si).clipped_fraction, manifest(si).eeg_std_uv);
    clear eeg_uv eeg_uv_cont fnirs_um old;
end

save(fullfile(output_dir, 'manifest_v2.mat'), 'manifest', '-v7');
fid = fopen(fullfile(output_dir, 'PREPROCESSING_V2.txt'), 'w');
fprintf(fid, ['HYGRIP EEG v2: continuous 1000->200 Hz resampling; robust 12-MAD clipping; CAR;\n' ...
    '12.5/25/37.5 Hz notch filters (Q=50); zero-phase 1-45 Hz Butterworth; then 0-20 s epoching.\n' ...
    'Stored EEG is treated as volts (despite the incorrect HDF milivolts attribute) and converted V->uV.\n' ...
    'Verified v1 HbO/HbR trials and labels are preserved unchanged. No EOG/EMG ICA was applied.\n']);
fclose(fid);
fprintf('Prepared HYGRIP EEG v2 in %s\n', output_dir);
end
