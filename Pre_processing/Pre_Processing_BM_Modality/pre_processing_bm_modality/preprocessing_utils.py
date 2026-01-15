from collections import defaultdict, deque
import glob
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple
import pathlib
import pandas as pd
from tqdm import tqdm

import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold


def process_dataset(
        full_dataset_path: str,
        all_subjects_dirs: List,
        pre_processing_cfg: Dict[str, Any],
        outputs_folder: str,
        seq_len: int = 5,
        overlap: float = 0.,
        frequency: int = 10,
        resample_freq: int = 10,
        use_sensors: Optional[List[str]] = None,
        borders: Optional[List[float]] = None,
):
    """
    Preprocesses the dataset with bio-measurements in Magic XRoom format.
    Saves the processed audio in the output directory.

    Args:
        full_dataset_path: the path to the full dataset
        all_subjects_dirs: a list containing all subdirectories which is assumed to be all the different subjects
        pre_processing_cfg: configutation for pre-processing
        outputs_folder: path to the outputs folder,
        seq_len: sequence length in seconds
        overlap: overlapping proportion between segments in [0, 1)
        frequency: frequency of the raw signal
        borders: list of float values defining the borders for each category
    Returns:
        train_split: a dictionary containing the 'files' and 'labels' for training
        val_split: a dictionary containing the 'files' and 'labels' for validation
        test_split: a dictionary containing the 'files' and 'labels' for testing
    """
    get_ssl = pre_processing_cfg["get_ssl"] if "get_ssl" in pre_processing_cfg else False
    get_stats = pre_processing_cfg["get_stats"] if "get_stats" in pre_processing_cfg else False


    # get the right function to use, and create path to save files to is doesnt exist
    self_functions = {
        "normalize": normalize,
        'standardize': standardize,
        'raw': no_preprocessing
    }

    preprocessing_to_apply = self_functions[pre_processing_cfg['process']]
    pathlib.Path(
        os.path.join(
            outputs_folder,
            pre_processing_cfg['process']
        )
    ).mkdir(parents=True, exist_ok=True)

    if get_ssl:
        pathlib.Path(
            os.path.join(
                outputs_folder,
                "ssl_" + pre_processing_cfg['process']
            )
        ).mkdir(parents=True, exist_ok=True)

    # go over each phase/split
    ovr_stats = []
    processed_files = []
    ssl_processed_files = []

    for subject_path in tqdm(all_subjects_dirs, desc=f"Preprocessing subject folders"):
        subject_path = os.path.join(full_dataset_path, subject_path)
        # format: data_collection_SESSION_SENSOR_.csv
        sessions = set([x.split("_")[2] for x in os.listdir(subject_path)])

        for session in sessions:
            processed_file_paths = []
            if get_ssl:
                processed_file_paths_ssl = []
            processed_file_labels = []
            processed_file_infos = []
            processed_file_games = []

            session_annot = glob.glob(os.path.join(subject_path, f"*{session}*PROGRESS_EVENT_.csv"))[0]
            session_bm = glob.glob(os.path.join(subject_path, f"*{session}*SHIMMER_.csv"))[0]

            # Assign available labels from annotations to bio-measurement data
            # to obtain dataframe with labeled signals corresponding to multiple levels (intervals)
            processed_session, stats, processed_session_ssl = process_session(
                session_bm,
                session_annot,
                subject_path,
                session=session,
                get_ssl=get_ssl,
                get_stats=get_stats,
                use_sensors=use_sensors,
                borders=borders
            )
            if get_stats:
                if stats:
                    ovr_stats.append(stats)

            # Segment each extracted level into shorter time windows
            # Each level (interval) will be split into multiple segments with the same length
            try:
                segmented_session, labels, infos, games = segment_processed_session(
                    processed_session,
                    seq_len,
                    overlap,
                    frequency=frequency
                )
            except ValueError as e:
                print(f"Error segmenting session: {str(e)}")
                segmented_session = None

            if segmented_session is not None:
                # apply pre-processing (e.g., normalization) for the whole session
                preprocessed_session = preprocessing_to_apply(segmented_session)

                for i, session_to_save in enumerate(preprocessed_session):
                    if games[i] is None:
                        print(f"Game is None for: subject path: {subject_path}, session: {session}, interval: {i}")
                        continue
                    # apply resampling if needed
                    if resample_freq != frequency:
                        session_to_save = resample_bm(session_to_save, frequency, resample_freq)

                    filepath = os.path.join(
                        outputs_folder,
                        pre_processing_cfg['process'],
                        f"{os.path.basename(subject_path)}_{session}_{i}_emotion_{labels[i]}_game_{games[i]}.npy"
                    )
                    np.save(filepath, session_to_save.astype(np.float32))

                    processed_file_paths.append(filepath.split(os.sep)[-1])
                    processed_file_labels.append(labels[i])
                    processed_file_infos.append(infos[i])
                    processed_file_games.append(games[i])
                
                for i, filepath in enumerate(processed_file_paths):
                    processed_files.append(
                        (
                            filepath,
                            os.path.basename(subject_path),
                            processed_file_labels[i],
                            processed_file_infos[i],
                            processed_file_games[i],
                            session,
                        )
                    )
            # repeat the processing for unlabeled ssl data
            if get_ssl:
                try:
                    segmented_session_ssl, ssl_games = segment_processed_session_ssl(
                        processed_session_ssl,
                        seq_len,
                        overlap,
                        frequency=frequency
                    )
                except ValueError:
                    segmented_session_ssl = None
                    ssl_games = None

                if segmented_session_ssl is not None:
                    # apply pre-processing (e.g., normalization) for the whole session
                    preprocessed_session_ssl = preprocessing_to_apply(segmented_session_ssl)

                    for i, session_to_save in enumerate(preprocessed_session_ssl):
                        if resample_freq != frequency:
                            session_to_save = resample_bm(session_to_save, frequency, resample_freq)
                        filepath = os.path.join(
                            outputs_folder,
                            "ssl_" + pre_processing_cfg['process'],
                            f"{os.path.basename(subject_path)}_{session}_{i}.npy"
                        )

                        np.save(filepath, session_to_save.astype(np.float32))

                        processed_file_paths_ssl.append(filepath.split(os.sep)[-1])

                        ssl_processed_files.append(
                            (
                                filepath.split(os.sep)[-1],
                                os.path.basename(subject_path),
                                None,
                                ssl_games[i],
                                session,
                            )
                        )

            if segmented_session is None:
                print(f"""Skipping subject {subject_path} session {session}.
                        Error in pre-processing labeled data: Not enough labeled data""")
            if get_ssl and segmented_session_ssl is None:
                print(f"""Skipping subject {subject_path} session {session}.
                        Error in pre-processing unlabeled data: Not enough unlabeled data""")

    df_processed = pd.DataFrame(processed_files, columns=["files", "subject", "labels", "infos", "game", "session"])
    df_ssl_processed = pd.DataFrame(ssl_processed_files, columns=["files", "subject", "labels", "game", "session"])

    # Filter out 0.5 values if configured
    filter_05 = pre_processing_cfg.get("filter_05", False)
    if filter_05:
        df_processed = df_processed[~df_processed["infos"].isin([0.5, 0.5000001])].copy()

    # Drop infos column as it should not be saved (cannot be used as features)
    df_processed = df_processed.drop(columns=["infos"])

    split_by = pre_processing_cfg.get("split_by", "subject")
    stratify = pre_processing_cfg.get("stratify", False) # Balance the train/val/test splits by split_by and labels
    game_split = pre_processing_cfg.get("game_split", None) # Which games to include in each split

    unique_feat_values = df_processed[split_by].unique()
    unique_feat_filtered = [feat_value for feat_value in unique_feat_values if feat_value is not None]

    if split_by == "game" and game_split:
        train_feat_value = game_split.get("train", [])
        val_feat_value = game_split.get("val", [])
        test_feat_value = game_split.get("test", [])
        
        configured_games = set(train_feat_value) | set(val_feat_value) | set(test_feat_value)
        missing_games = set(unique_feat_filtered) - configured_games
        if missing_games:
            print(f"Warning: The following games found in the dataset are not included in the split configuration: {missing_games}")
            
    elif stratify:
        balanced = False
        n_tries = 0
        max_tries = 10
        
        while not balanced and n_tries < max_tries:
            # Create a group-level dataset: one row per group with aggregated label distribution
            group_data = []
            for group_val in unique_feat_filtered:
                group_df = df_processed[df_processed[split_by] == group_val]
                label_counts = group_df["labels"].value_counts().to_dict()
                # Use majority label for stratification
                majority_label = group_df["labels"].mode()[0]
                group_data.append({
                    'group': group_val,
                    'majority_label': majority_label,
                    'label_counts': label_counts
                })
            
            group_df_agg = pd.DataFrame(group_data)
            
            # Split is made at the group level
            n_groups = len(unique_feat_filtered)
            n_splits = max(3, min(7, n_groups))
            
            X_groups = np.zeros(len(group_df_agg))
            y_groups = group_df_agg["majority_label"]
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            
            all_folds = list(skf.split(X_groups, y_groups))
            
            # Assign folds to splits
            if n_splits == 3:
                test_idx = all_folds[0][1]
                val_idx = all_folds[1][1]
                train_idx = all_folds[2][1]
            else:
                test_idx = all_folds[0][1]
                val_idx = all_folds[1][1]
                train_idx = np.concatenate([all_folds[i][1] for i in range(2, n_splits)])
            
            # Get the actual group values
            test_feat_value = group_df_agg.iloc[test_idx]['group'].values
            val_feat_value = group_df_agg.iloc[val_idx]['group'].values
            train_feat_value = group_df_agg.iloc[train_idx]['group'].values
            
            # Verify no overlap between splits
            assert len(set(train_feat_value) & set(val_feat_value)) == 0, "Train/Val overlap!"
            assert len(set(train_feat_value) & set(test_feat_value)) == 0, "Train/Test overlap!"
            assert len(set(val_feat_value) & set(test_feat_value)) == 0, "Val/Test overlap!"
            
            # Verify all classes are present in each split
            train_classes = df_processed[df_processed[split_by].isin(train_feat_value)]["labels"].unique()
            val_classes = df_processed[df_processed[split_by].isin(val_feat_value)]["labels"].unique()
            test_classes = df_processed[df_processed[split_by].isin(test_feat_value)]["labels"].unique()
            
            print(f"Attempt {n_tries + 1}:")
            print(f"  Train: {len(train_feat_value)} groups, classes: {sorted(train_classes)}")
            print(f"  Val: {len(val_feat_value)} groups, classes: {sorted(val_classes)}")
            print(f"  Test: {len(test_feat_value)} groups, classes: {sorted(test_classes)}")
            
            all_classes = df_processed["labels"].unique()
            if len(test_classes) < len(all_classes):
                print(f"  WARNING: Test set is missing classes: {set(all_classes) - set(test_classes)}")
            if len(val_classes) < len(all_classes):
                print(f"  WARNING: Val set is missing classes: {set(all_classes) - set(val_classes)}")
            
            balanced = len(train_classes) == len(val_classes) == len(test_classes) == len(all_classes)
            n_tries += 1
        
        if balanced:
            print(f"\n Balanced split found after {n_tries} tries")
        else:
            print(f"\n Could not find balanced split after {max_tries} tries, current split may have some class imbalance")
    else:
        test_feat_value = np.random.choice(unique_feat_filtered, max(1, int(0.1 * len(unique_feat_filtered))), replace=False)
        no_test = [feat_value for feat_value in unique_feat_filtered if feat_value not in test_feat_value]
        val_feat_value = np.random.choice(no_test, max(1, int(0.1 * len(unique_feat_filtered))), replace=False)
        train_feat_value = [
            feat_value for feat_value in unique_feat_filtered 
            if feat_value not in test_feat_value and feat_value not in val_feat_value
        ]

    # Create splits
    train_split_df = df_processed[df_processed[split_by].isin(train_feat_value)]
    val_split_df = df_processed[df_processed[split_by].isin(val_feat_value)]
    test_split_df = df_processed[df_processed[split_by].isin(test_feat_value)]

    ssl_train_split_df = df_ssl_processed[
        df_ssl_processed[split_by].isin(train_feat_value) | df_ssl_processed[split_by].isna()
    ]
    ssl_val_split_df = df_ssl_processed[
        df_ssl_processed[split_by].isin(val_feat_value)
    ]
    ssl_test_split_df = df_ssl_processed[
        df_ssl_processed[split_by].isin(test_feat_value)
    ]

    train_split = train_split_df.to_dict(orient="records")
    val_split = val_split_df.to_dict(orient="records")
    test_split = test_split_df.to_dict(orient="records")
    ssl_train_split = ssl_train_split_df.to_dict(orient="records")
    ssl_val_split = ssl_val_split_df.to_dict(orient="records")
    ssl_test_split = ssl_test_split_df.to_dict(orient="records")

    return (
        train_split,
        val_split,
        test_split,
        ovr_stats if get_stats else None,
        ssl_train_split if get_ssl else [],
        ssl_val_split if get_ssl else [],
        ssl_test_split if get_ssl else [],
    )


def process_session(
    session_data_file: str,
    session_annot_file: str,
    subject: str,
    session: str,
    threshold: float = 30,
    offset_hours_data: int = 1,
    get_ssl: bool = False,
    get_stats: bool = False,
    use_sensors: Optional[List[str]] = None,
    cont_to_cat: bool = True,
    borders: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts session data from Shimmer and Progress Events from Magic XRoom and assigns labels to sensor recordings.
    Collects summary about the session.

    Args:
        session_data_file: path to bio-measurements recordings from the session
        session_annot_file: path to annotations
        subject: subject ID
        session: session ID
        threshold: threshold in seconds to discard irrelevant labels and levels,
            i.e. if the time passed between annotation and the latest completed/failed level is higher than
            this threshold, both annotation and level are discarded
        offset_hours_data: delay in Shimmer data recording in hours caused by different time-zones
        get_ssl: return unlabeled dataframe together with annotated dataframe
        get_stats: flag for generating stats csv (for labeled data)
        borders: list of float values defining the borders for each category

    Returns:
        labeled_data: dataframe consisting labeled sensor recordings
        stats: dictionary with stats describing the input session
    """
    # Timestamp format: C# ticks
    annotations = pd.read_csv(session_annot_file)
    # Check if file has headers by looking at first row
    if not any(col in annotations.columns for col in ["timestamp", "event_type", "info"]):
        # Reset the DataFrame with proper column names
        annotations = pd.read_csv(session_annot_file, names=["timestamp", "event_type", "info"])
        
    annotations["timestamp_dt"] = (
        annotations["timestamp"]
        .apply(lambda x: datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=x // 10))
    )
    annotations = (
        annotations
        .sort_values(by="timestamp_dt")
        .reset_index()
    )

    # Timestamp format: Unix
    data = pd.read_csv(session_data_file)
    if use_sensors is not None:
        save_cols = ["timestamp"]
        save_cols.extend(use_sensors)
        data = data[save_cols]
    # Older version of Magic XRoom collects Shimmer internal UNIX timestamp as 'timestamp' column
    try:
        data["timestamp_dt"] = (
            data["timestamp"]
            .apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        )
    # Current version of Magic XRoom uses C# timestamp as 'timestamp' column
    #   and internal UNIX timestamps as 'timestamp_int' column
    except (ValueError, OSError):
        data["timestamp_dt"] = (
            data["timestamp"]
            .apply(lambda x: datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=x // 10))
        )
        offset_hours_data = 0
    if offset_hours_data >= 1:
        data['timestamp_dt'] += pd.Timedelta(hours=offset_hours_data)
    data = data.drop_duplicates()
    data["interval_num"] = np.nan
    data["label"] = np.nan
    data = (
        data
        .sort_values(by="timestamp_dt")
        .reset_index()
    )

    stack_level_ts = []
    labeled_intervals = deque()

    current_game = None
    # iterate through annotations file to assign labels to level timestamps
    for _, row in annotations.iterrows():
        event_type = row["event_type"]
        info = row["info"]
        if event_type == "SCENARIO_STARTED":
            current_game = info
        if event_type == "LEVEL_STARTED":
            # it is not expected to have LEVEL_STARTED two times in a row
            start_ts = row["timestamp_dt"]

        elif event_type in ["LEVEL_COMPLETED", "LEVEL_FAILED"]:
            # save interval to stack if level_started
            if 'start_ts' not in locals():
                continue
            stack_level_ts.append((start_ts, row['timestamp_dt']))

        elif event_type in ["BORED", "ENGAGED", "FRUSTRATED", "SKIP", "FEEDBACK_RECEIVED"]:
            last_finished_level_start, last_finished_level_end = stack_level_ts.pop() if (
                stack_level_ts
            ) else (None, None)
            while stack_level_ts:
                prev_start, prev_end = stack_level_ts.pop()
                if (prev_end - last_finished_level_start).total_seconds() < threshold:
                    last_finished_level_start = prev_start
            # assign label to the latest level interval (from stack) if time difference is not larger than a threhsold
            if (
                event_type != "SKIP" and
                last_finished_level_end is not None and
                (row["timestamp_dt"] - last_finished_level_end).total_seconds() < threshold
            ):
                if event_type != "FEEDBACK_RECEIVED":
                    label = event_type
                    info_val = None
                else:
                    label = continious_to_categorical(info, borders=borders) if cont_to_cat else info
                    info_val = float(info)
                labeled_intervals.append(
                    (
                        last_finished_level_start,
                        last_finished_level_end,
                        label,
                        info_val,
                        row["timestamp_dt"],
                        current_game
                    )
                )

    sensor_first_entry = min(data['timestamp_dt'])
    sensor_last_entry = max(data['timestamp_dt'])
    progress_event_first_entry = min(annotations['timestamp_dt']) if annotations.shape[0] > 0 else None
    progress_event_last_entry = max(annotations['timestamp_dt']) if annotations.shape[0] > 0 else None

    start, end, label, info_val, _, game_name = labeled_intervals.popleft() if labeled_intervals else (None, None, None, None, None, None)
    interval_num = 1

    data["game_name"] = None
    data["info"] = 0.0
    # iterate through data (bio-measurements) to assign labels based on intervals
    if None not in [start, end]:
        for idx, row in data.iterrows():
            if start <= row["timestamp_dt"] <= end:
                data.at[idx, "label"] = label
                data.at[idx, "info"] = info_val if info_val is not None else 0.0
                data.at[idx, "interval_num"] = interval_num
                data.at[idx, "game_name"] = game_name
            elif end < row['timestamp_dt']:
                if not labeled_intervals:
                    break
                start, end, label, info_val, _, game_name = labeled_intervals.popleft()
                interval_num += 1

    # query labeled data
    labeled_data = data[~data["label"].isna()]
    stats = {}
    if get_stats and not labeled_data.empty:
        # compute length of recorded labeled data and each emotion
        min_max_intervals = (
            labeled_data[["interval_num", "timestamp_dt", "label"]]
            .groupby(by="interval_num")
            .agg([np.min, np.max])
            .reset_index(drop=True)
        )
        min_max_intervals.columns = min_max_intervals.columns.map('_'.join).str.strip('_')
        min_max_intervals = min_max_intervals[["timestamp_dt_min", "timestamp_dt_max", "label_min"]]
        min_max_intervals["interval_length"] = (
            min_max_intervals["timestamp_dt_max"] - min_max_intervals["timestamp_dt_min"]
        )
        min_max_intervals["interval_length"] = min_max_intervals["interval_length"].dt.total_seconds()
        length_of_labeled_data = min_max_intervals["interval_length"].sum()

        length_per_min_max_intervals = (
            min_max_intervals
            .groupby(by=["label_min"])
            .sum(numeric_only=True)
            .reset_index()
        )

        lengths = {
            "length_seconds_BORED": pd.Timedelta(0),
            "length_seconds_ENGAGED": pd.Timedelta(0),
            "length_seconds_FRUSTRATED": pd.Timedelta(0)
        }

        for _, row in length_per_min_max_intervals.iterrows():
            lengths[f"length_seconds_{row['label_min']}"] = row["interval_length"]

        stats = {
            "subject": os.path.basename(subject),
            "session": session,
            "sensor_session_length": sensor_last_entry - sensor_first_entry,
            "progress_event_length": progress_event_last_entry - progress_event_first_entry,
            "avg_actual_frequency": data.shape[0] / int((sensor_last_entry - sensor_first_entry).total_seconds()),
            "progress_event_first_entry": progress_event_first_entry,
            "progress_event_last_entry": progress_event_last_entry,
            "sensor_first_entry": sensor_first_entry,
            "sensor_last_entry": sensor_last_entry,
            "labeled_sensor_data_pct": round(labeled_data.shape[0] / data.shape[0], 2),
            "num_labeled_intervals_seconds": len(labeled_data["interval_num"].unique()),
            "length_labeled_intervals": length_of_labeled_data,
        }
        stats = {**stats, **lengths}

    if get_ssl and not labeled_data.empty:
        ssl_data = data.merge(labeled_data, on=["index"], how="left", suffixes=("", "_r"))
        columns_left_drop = ["label", "game_name", "info"]
        columns_right_keep = ["label_r", "game_name_r"]
        columns_right_drop = [col for col in ssl_data.columns if col.endswith("_r") and col not in columns_right_keep]
        ssl_data = ssl_data.drop(columns=columns_left_drop + columns_right_drop)
        ssl_data = ssl_data.rename(columns={"label_r": "label", "game_name_r": "game_name"})
        # make sure that no data is discarded
        # assert ssl_data[~ssl_data["label"].isna()].equals(labeled_data)
        assert len(ssl_data) == len(data)
    elif get_ssl:
        ssl_data = data
    else:
        ssl_data = None

    return (
        labeled_data,
        stats if get_stats else None,
        ssl_data
    )


def segment_processed_session(
    session_df: pd.DataFrame,
    seq_len: int,
    overlap: float,
    frequency: float = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Segmenting processed sessions into time windows of the provided sequence length in seconds using labeled intervals

    Args:
        session_df: Dataframe obtained after calling process_session()
        seq_len: lengths of sequences (time windows) in seconds
        overlap: proportion of overlap between time windows in [0, 1)
        frequency: (expected) frequency of the signal
    """
    window_length = int(seq_len * frequency)
    intervals = session_df["interval_num"].unique()
    segmented_session = []
    labels = []
    infos = []
    games = []
    for interval in intervals:
        interval_data = session_df[session_df["interval_num"] == interval]
        unique_labels = interval_data["label"].unique()
        unique_infos = interval_data["info"].unique()
        unique_games = interval_data["game_name"].unique()
        if len(unique_labels) > 1:
            raise ValueError("Found multiple labels per interval")
        label = unique_labels[0]
        info = unique_infos[0]
        game = unique_games[0]
        drop_cols = [
            "index",
            "timestamp",
            "updated_timestamp",
            "timestamp_dt",
            "label",
            "info",
            "interval_num",
            "game_name"
        ]
        interval_data_sensors = np.array(
            interval_data
            .drop([x for x in drop_cols if x in interval_data.columns], axis=1)
        )

        for i in range(0, len(interval_data_sensors) - window_length, int(window_length * (1 - overlap))):
            curr_window = interval_data_sensors[i: i + window_length]
            segmented_session.append(curr_window)
            labels.append(label)
            infos.append(info)
            games.append(game)
    return np.stack(segmented_session), labels, infos, games


def segment_processed_session_ssl(
    session_df: pd.DataFrame,
    seq_len: int,
    overlap: float,
    frequency: float = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Segmenting processed sessions into time windows of the provided sequence length in seconds for ssl data

    Args:
        session_df: Dataframe obtained after calling process_session()
        seq_len: lengths of sequences (time windows) in seconds
        overlap: proportion of overlap between time windows in [0, 1)
        frequency: (expected) frequency of the signal
    """
    window_length = int(seq_len * frequency)
    segmented_session = []
    drop_cols = [
        "index",
        "timestamp",
        "updated_timestamp",
        "timestamp_dt",
        "label",
        "info",
        "interval_num",
        "game_name"
    ]
    session_data_sensors = np.array(
        session_df
        .drop([x for x in drop_cols if x in session_df.columns], axis=1)
    )
    
    # Extract games if available, otherwise fill with None
    if "game_name" in session_df.columns:
        session_games = session_df["game_name"].values
    else:
        session_games = np.array([None] * len(session_df))

    games = []
    for i in range(0, len(session_data_sensors) - window_length, int(window_length * (1 - overlap))):
        curr_window = session_data_sensors[i: i + window_length]
        segmented_session.append(curr_window)
        
        # Get game for this window. Using middle sample to be safe against boundary conditions
        mid_idx = i + window_length // 2
        games.append(session_games[mid_idx])

    return np.stack(segmented_session), games


def normalize(bm_segments):
    """
    normalize: transformed into a range between -1 and 1 by normalization for each speaker (min-max scaling)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    min_channel = stacked_segments.min(axis=0)
    max_channel = stacked_segments.max(axis=0)

    segments_min_max = (bm_segments - min_channel) / (max_channel - min_channel)

    return segments_min_max


def standardize(bm_segments: np.ndarray):
    """
    z-normalization to zero mean and unit variance for each segment with bio-measurements

    Args:
        bm_segment: 3D-array containing bio-measurement signals from one session (multiple segments per session)

    Returns:
        standardized_signal: a list containing the standardized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    mean_channel = stacked_segments.mean(axis=0)
    std_channel = stacked_segments.std(axis=0)

    bm_z_normalized = (bm_segments - mean_channel) / std_channel

    return bm_z_normalized


def no_preprocessing(bm_segment):
    """
    No pre-processing applied

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return bm_segment


def resample_bm(
        bm_segment: np.ndarray,
        sample_rate: int,
        target_rate: int
):
    """
    Resample stacked signals to a target frequency

    Args:
        bm_segment: 3D numpy array with the bio-measurement data to resample
        sample_rate: the sample rate of the original signal
        target_rate: the target sample rate
    Returns:
        resampled: resampled signals
    """
    number_of_samples = round(len(bm_segment) * float(target_rate) / sample_rate)
    resampled = scipy.signal.resample(bm_segment, number_of_samples, axis=0)
    return resampled


def continious_to_categorical(info: str, categories=["BORED", "ENGAGED", "FRUSTRATED"], borders=None) -> str:
    """
    Maps a continuous value in the range [0, 1] to a discrete category.

    Args:
        info: A string containing a float value between 0 and 1.
        categories: Categories to map continuous values to.
        borders: A list of float values defining the borders for each category.
                 Must be in ascending order and have len(categories) - 1 elements.

    """
    num_categories = len(categories)
    try:
        value = float(info)
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")

        if borders is None:
            category_size = 1. / num_categories
            category_index = min(int(value // category_size), num_categories - 1)
        elif borders[0] == borders[1]: # Binary classification
            category_index = 0 if value < borders[0] else 2
        else:
            if len(borders) != num_categories - 1:
                raise ValueError("Borders list must have len(categories) - 1 elements.")
            category_index = next((i for i, border in enumerate(borders) if value < border), num_categories - 1)

        category = categories[category_index]

        return str(category)
    except ValueError:
        print(f"Invalid input: {info}. Expected a float between 0 and 1.")
        return "INVALID"
