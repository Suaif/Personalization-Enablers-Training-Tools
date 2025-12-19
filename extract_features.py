import os
import numpy as np
import neurokit2 as nk
import warnings
import shutil


def generate_and_save_features(
    data_dir: str,
    output_dir: str,
    sampling_rate=10,
    signals=["gsr"]
):
    """
    Generate features for all files in the data directory and save them.

    signals: list of ["gsr", "ppg"]
    """

    print(f"Generating features for {data_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"{output_dir} already exists. Removing all files")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    print("Extracting features...")
    
    n = 0
    total_gsr_warnings = 0
    total_ppg_warnings = 0

    for file in os.listdir(data_dir):
        if not file.endswith(".npy"):
            continue

        data = np.load(os.path.join(data_dir, file))

        features = {}

        # Assume: data[:, 0] = GSR and data[:, 1] = PPG
        if "gsr" in signals:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gsr_features = extract_eda_features_from_raw(
                    data[:, 0],
                    sampling_rate=sampling_rate
                )

                gsr_warnings = [
                    warn for warn in w
                    if "All-NaN slice encountered" in str(warn.message)
                ]
                total_gsr_warnings += len(gsr_warnings)

            features.update(gsr_features)

        if "ppg" in signals:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ppg_features = extract_ppg_features_from_raw(
                    data[:, 1],
                    sampling_rate=17
                )

                ppg_warnings = [
                    warn for warn in w
                    if "Returning empty vector" in str(warn.message)
                ]
                total_ppg_warnings += len(ppg_warnings)

            features.update(ppg_features)

        np.save(
            os.path.join(output_dir, file),
            np.array(list(features.values()))
        )

        n += 1

    print(f"Generated features for {n} files.")
    print(f"Total GSR warnings: {total_gsr_warnings}")
    print(f"Total PPG warnings: {total_ppg_warnings}")


def extract_eda_features_from_raw(
    gsr_window,
    sampling_rate=10
):
    """
    Extract basic EDA features from a raw GSR window.

    Parameters
    ----------
    gsr_window : array-like, shape (n_samples,)
        Raw GSR signal (e.g. 5 seconds at 10 Hz)
    sampling_rate : int or float
        Sampling rate in Hz

    Returns
    -------
    features : dict
        Flat dictionary of scalar features
    """

    features = {}

    # ---------------------
    # NeuroKit processing
    # ---------------------
    eda_signals, eda_info = nk.eda_process(
        gsr_window,
        sampling_rate=sampling_rate
    )

    tonic = eda_signals["EDA_Tonic"].values
    phasic = eda_signals["EDA_Phasic"].values

    n_samples = len(phasic)
    duration = n_samples / sampling_rate

    # ---------------------
    # PHASIC FEATURES
    # ---------------------
    features["gsr_phasic_mean"] = np.mean(phasic)
    features["gsr_phasic_std"] = np.std(phasic)
    features["gsr_phasic_max"] = np.max(phasic)
    features["gsr_phasic_rms"] = np.sqrt(np.mean(phasic ** 2))
    features["gsr_phasic_auc"] = np.sum(phasic) / n_samples

    d_phasic = np.diff(phasic)
    features["gsr_phasic_d1_mean_abs"] = np.mean(np.abs(d_phasic))
    features["gsr_phasic_d1_max_abs"] = np.max(np.abs(d_phasic))

    # ---------------------
    # TONIC FEATURES
    # ---------------------
    features["gsr_tonic_mean"] = np.mean(tonic)
    features["gsr_tonic_std"] = np.std(tonic)
    features["gsr_tonic_min"] = np.min(tonic)
    features["gsr_tonic_max"] = np.max(tonic)

    t = np.arange(n_samples) / sampling_rate
    features["gsr_tonic_slope"] = np.polyfit(t, tonic, 1)[0]
    features["gsr_tonic_delta"] = tonic[-1] - tonic[0]

    # ---------------------
    # SCR EVENT FEATURES
    # ---------------------
    scr_amplitudes = np.asarray(eda_info.get("SCR_Amplitude", []))
    scr_rise_times = np.asarray(eda_info.get("SCR_RiseTime", []))

    # Drop NaNs explicitly
    scr_amplitudes = scr_amplitudes[~np.isnan(scr_amplitudes)]
    scr_rise_times = scr_rise_times[~np.isnan(scr_rise_times)]

    n_scr = len(scr_amplitudes)

    features["gsr_scr_count"] = n_scr
    features["gsr_scr_rate"] = n_scr / duration

    if n_scr > 0:
        features["gsr_scr_amp_mean"] = np.mean(scr_amplitudes)
        features["gsr_scr_amp_max"] = np.max(scr_amplitudes)
        features["gsr_scr_amp_sum"] = np.sum(scr_amplitudes)
        features["gsr_scr_risetime_mean"] = np.mean(scr_rise_times)
        features["gsr_scr_risetime_std"] = np.std(scr_rise_times)
    else:
        # Explicit zeros encode "no arousal"
        features["gsr_scr_amp_mean"] = 0.0
        features["gsr_scr_amp_max"] = 0.0
        features["gsr_scr_amp_sum"] = 0.0
        features["gsr_scr_risetime_mean"] = 0.0
        features["gsr_scr_risetime_std"] = 0.0

    return _sanitize_features(features)

def extract_ppg_features_from_raw(
    ppg_window,
    sampling_rate=10
):
    """
    Extract basic PPG features from a raw PPG window.
    """

    features = {}

    ppg_signals, ppg_info = nk.ppg_process(
        ppg_window,
        sampling_rate=sampling_rate
    )

    signal = ppg_signals["PPG_Clean"].values

    # Basic signal stats
    features["ppg_mean"] = np.mean(signal)
    features["ppg_std"] = np.std(signal)
    features["ppg_rms"] = np.sqrt(np.mean(signal ** 2))
    features["ppg_auc"] = np.sum(signal) / len(signal)

    # Dynamics
    d_signal = np.diff(signal)
    features["ppg_d1_mean_abs"] = np.mean(np.abs(d_signal))
    features["ppg_d1_max_abs"] = np.max(np.abs(d_signal))

    # Heart rate (if detected)
    hr = ppg_signals.get("PPG_Rate", None)

    if hr is not None:
        hr = hr.values
        hr = hr[~np.isnan(hr)]

        if len(hr) > 0:
            features["ppg_hr_mean"] = np.mean(hr)
            features["ppg_hr_std"] = np.std(hr)
        else:
            features["ppg_hr_mean"] = 0.0
            features["ppg_hr_std"] = 0.0
    else:
        features["ppg_hr_mean"] = 0.0
        features["ppg_hr_std"] = 0.0

    return _sanitize_features(features)


def _sanitize_features(features, fill_value=0.0):
    """
    Replace NaN / inf values with a safe constant.
    """
    clean = {}
    for k, v in features.items():
        if np.isnan(v):
            clean[k] = fill_value
        else:
            clean[k] = float(v)

    return clean

if __name__ == "__main__":
    
    base_dir = "C:/Users/ismas/Documents/X2Learn/Personalization-Enablers-Training-Tools"
    data_dir = os.path.join(base_dir, "outputs", "XRoom", "shimmer", "standardize")
    output_dir = os.path.join(base_dir, "outputs", "XRoom", "shimmer", "eda_features")
    generate_and_save_features(data_dir, output_dir)