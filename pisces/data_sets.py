# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_data_sets.ipynb.

# %% auto 0
__all__ = ['LOG_LEVEL', 'vec_to_WLDM', 'SimplifiablePrefixTree', 'IdExtractor', 'DataSetObject', 'psg_to_sleep_wake', 'to_WLDM',
           'psg_to_WLDM', 'ModelOutputType', 'PSGType', 'ModelInput', 'ModelInput1D', 'ModelInputSpectrogram',
           'get_sample_weights', 'mask_psg_from_accel', 'apply_gausian_filter', 'fill_gaps_in_accelerometer_data',
           'DataProcessor']

# %% ../nbs/01_data_sets.ipynb 4
import os
import re
import logging
import warnings
import numpy as np
import polars as pl
from pathlib import Path
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from collections import defaultdict
from typing import Dict, List, Tuple
from .mads_olsen_support import *
from typing import DefaultDict, Iterable
from scipy.ndimage import gaussian_filter1d
from .utils import determine_header_rows_and_delimiter

# %% ../nbs/01_data_sets.ipynb 6
class SimplifiablePrefixTree:
    """
    A standard prefix tree with the ability to "simplify" itself by combining nodes with only one child.
    These also have the ability to "flatten" themselves, which means to convert all nodes at and below a certain depth into leaves on the most recent ancestor of that depth.
    """
    def __init__(self, delimiter: str = "", # The delimiter to use when splitting words into characters. If empty, the words are treated as sequences of characters.
                 key: str = "", # The key of the current node in its parent's `.children` dictionary. If empty, the node is (likely) the root of the tree.
                 ):
        """
        key : str
            The key of the current node in its parent's `.children` dictionary. If empty, the node is (likely) the root of the tree.
        children : Dict[str, SimplifiablePrefixTree]
            The children of the current node, stored in a dictionary with the keys being the children's keys.
        is_end_of_word : bool
            Whether the current node is the end of a word. Basically, is this a leaf node?
        delimiter : str
            The delimiter to use when splitting words into characters. If empty, the words are treated as sequences of characters.
        print_spacer : str
            The string to use to indent the printed tree.
        """
        self.key = key
        self.children: Dict[str, SimplifiablePrefixTree] = {}
        self.is_end_of_word = False
        self.delimiter = delimiter
        self.print_spacer = "++"
    
    def chars_from(self, word: str):
        """
        Splits a word into characters, using the `delimiter` attribute as the delimiter.
        """
        return word.split(self.delimiter) if self.delimiter else word

    def insert(self, word: str):
        """
        Inserts a word into the tree. If the word is already in the tree, nothing happens.
        """
        node = self
        for char in self.chars_from(word):
            if char not in node.children:
                node.children[char] = SimplifiablePrefixTree(self.delimiter, key=char)
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Searches for a word in the tree.
        """
        node = self
        for char in self.chars_from(word):
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def simplified(self) -> 'SimplifiablePrefixTree':
        """
        Returns a simplified copy of the tree. The original tree is not modified.
        """
        self_copy = deepcopy(self)
        return self_copy.simplify()
    
    def simplify(self):
        """
        Simplifies the tree in place.
        """
        if len(self.children) == 1 and not self.is_end_of_word:
            child_key = list(self.children.keys())[0]
            self.key += child_key
            self.children = self.children[child_key].children
            self.simplify()
        else:
            current_keys = list(self.children.keys())
            for key in current_keys:
                child = self.children.pop(key)
                child.simplify()
                self.children[child.key] = child
        return self
    
    def reversed(self) -> 'SimplifiablePrefixTree':
        """
        Returns a reversed copy of the tree, except with with `node.key` reversed versus the node in `self.children`. The original tree is not modified.
        """
        rev_self = SimplifiablePrefixTree(self.delimiter, key=self.key[::-1])
        rev_self.children = {k[::-1]: v.reversed() for k, v in self.children.items()}
        return rev_self
    
    def flattened(self, max_depth: int = 1) -> 'SimplifiablePrefixTree':
        """
        Returns a Tree identical to `self` up to the given depth, but with all nodes at + below `max_depth` converted into leaves on the most recent ancestor of depth `max_depth - 1`.
        """
        flat_self = SimplifiablePrefixTree(self.delimiter, key=self.key)
        if max_depth == 0:
            if not self.is_end_of_word:
                warnings.warn(f"max_depth is 0, but {self.key} is not a leaf.")
            return flat_self
        if max_depth == 1:
            for k, v in self.children.items():
                if v.is_end_of_word:
                    flat_self.children[k] = SimplifiablePrefixTree(self.delimiter, key=k)
                else:
                    # flattened_children = v._pushdown()
                    for flattened_child in v._pushdown():
                        flat_self.children[flattened_child.key] = flattened_child
        else:
            for k, v in self.children.items():
                flat_self.children[k] = v.flattened(max_depth - 1)
        return flat_self
    
    def _pushdown(self) -> List['SimplifiablePrefixTree']:
        """
        Returns a list corresponding to the children of `self`, with `self.key` prefixed to each child's key.
        """
        pushed_down = [
            c
            for k in self.children.values()
            for c in k._pushdown()
        ]
        for i in range(len(pushed_down)):
            pushed_down[i].key = self.key + self.delimiter + pushed_down[i].key

        if not pushed_down:
            return [SimplifiablePrefixTree(self.delimiter, key=self.key)]
        else:
            return pushed_down
            

    def __str__(self):
        # prints .children recursively with indentation
        return self.key + "\n" + self.print_tree()

    def print_tree(self, indent=0) -> str:
        result = ""
        for key, child in self.children.items():
            result +=  self.print_spacer * indent + "( " + child.key + "\n"
            result += SimplifiablePrefixTree.print_tree(child, indent + 1)
        return result


class IdExtractor(SimplifiablePrefixTree):
    """
    Class extending the prefix trees that incorporates the algorithm for extracting IDs from a list of file names. The algorithm is somewhat oblique, so it's better to just use the `extract_ids` method versus trying to use the prfix trees directly at the call site.
    
    The algorithm is based on the assumption that the IDs are the same across all file names, but that the file names may have different suffixes. The algorithm reverses the file names, inserts them into the tree, and then simplifes and flattens that tree in order to find the IDs as leaves of that simplified tree.

    1. Insert the file name string into the tree, but with each string **reversed**.
    2. Simplify the tree, combining nodes with only one child.
    3. There may be unexpected suffix matches for these IDs, so we flatten the tree to depth 1, meaning all children of the root are combined to make leaves.
    4. The leaves are the IDs we want to extract. However, we must reverse these leaf keys to get the original IDs, since we reversed the file names in step 1.

    TODO:
    * If we want to find IDs for files with differing prefixes instead, we should instead insert the file names NOT reversed and then NOT reverse in the last step.

    * To handle IDs that appear in the middle of file names, we can use both methods to come up with a list of potential IDs based on prefix and suffix, then figure out the "intersection" of those lists. (Maybe using another prefix tree?)

    """
    def __init__(self, delimiter: str = "", key: str = ""):
        super().__init__(delimiter, key)

    def extract_ids(self, files: List[str]) -> List[str]:
        for file in files:
            self.insert(file[::-1])
        return sorted([
            c.key for c in self
                .prefix_flattened()
                .children
                .values()
        ])
    
    def prefix_flattened(self) -> 'IdExtractor':
        return self.simplified().flattened(1).reversed()
    

# %% ../nbs/01_data_sets.ipynb 11
LOG_LEVEL = logging.INFO

class DataSetObject:
    FEATURE_PREFIX = "cleaned_"

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.ids: List[str] = []

        # keeps track of the files for each feature and user
        self._feature_map: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
        self._feature_cache: DefaultDict[str, Dict[str, pl.DataFrame]] = defaultdict(dict)
    
    @property
    def features(self) -> List[str]:
        return list(self._feature_map.keys())
    
    def __str__(self):
        return f"{self.name}: {self.path}"

    def get_feature_data(self, feature: str, id: str) -> pl.DataFrame | None:
        if feature not in self.features:
            warnings.warn(f"Feature {feature} not found in {self.name}. Returning None.")
            return None
        if id not in self.ids:
            warnings.warn(f"ID {id} not found in {self.name}")
            return None
        if (df := self._feature_cache[feature].get(id)) is None:
            file = self.get_filename(feature, id)
            if not file:
                return None
            self.logger.debug(f"Loading {file}")
            try:
                n_rows, delimiter = determine_header_rows_and_delimiter(file)
                # self.logger.debug(f"n_rows: {n_rows}, delimiter: {delimiter}")
                df = pl.read_csv(file, has_header=True if n_rows > 0 else False,
                                 skip_rows=max(n_rows-1, 0), 
                                 separator=delimiter)
            except Exception as e:
                warnings.warn(f"Error reading {file}:\n{e}")
                return None
            # sort by time when loading
            df.sort(df.columns[0])
            self._feature_cache[feature][id] = df
        return df

    def get_filename(self, feature: str, id: str) -> Path | None:
        feature_ids = self._feature_map.get(feature)
        if feature_ids is None:
            # raise ValueError(f"Feature {feature_ids} not found in {self.name}")
            print(f"Feature {feature_ids} not found in {self.name}")
            return None
        file = feature_ids.get(id)
        if file is None:
            # raise ValueError
            print(f"ID {id} not found in {self.name}")
            return None
        return self.get_feature_path(feature)\
            .joinpath(file)
    
    def get_feature_path(self, feature: str) -> Path:
        return self.path.joinpath(self.FEATURE_PREFIX + feature)
    
    def _extract_ids(self, files: List[str]) -> List[str]:
        return IdExtractor().extract_ids(files)
    
    def add_feature_files(self, feature: str, files: Iterable[str]):
        if feature not in self.features:
            self.logger.debug(f"Adding feature {feature} to {self.name}")
            self._feature_map[feature] = {}
        # use a set for automatic deduping
        deduped_ids = set(self.ids)
        extracted_ids = sorted(self._extract_ids(files))
        files = sorted(list(files))
        # print('# extracted_ids:', len(extracted_ids))
        for id, file in zip(extracted_ids, files):
            # print('adding data for id:', id, 'file:', file)
            self._feature_map[feature][id] = file
            # set.add only adds the value if it's not already in the set
            deduped_ids.add(id)
        self.ids = sorted(list(deduped_ids))
    
    def get_feature_files(self, feature: str) -> Dict[str, str]:
        return {k: v for k, v in self._feature_map[feature].items()}
    
    def get_id_files(self, id: str) -> Dict[str, str]:
        return {k: v[id] for k, v in self._feature_map.items()}
    
    def load_feature_data(self, feature: str | None, id: str | None) -> Dict[str, np.ndarray]:
        if feature not in self.features:
            raise ValueError(f"Feature {feature} not found in {self.name}")
    
    @classmethod
    def find_data_sets(cls, root: str | Path) -> Dict[str, 'DataSetObject']:
        set_dir_regex = r".*" + cls.FEATURE_PREFIX + r"(.+)"
        # this regex matches the feature directory name and the data set name
        # but doesn't work on Windows (? maybe, cant test) because of the forward slashes
        feature_dir_regex = r".*/(.+)/" + cls.FEATURE_PREFIX + r"(.+)"

        data_sets: Dict[str, DataSetObject] = {}
        for root, dirs, files in os.walk(root, followlinks=True):
            # check to see if the root is a feature directory,
            # if it is, add that feature data to the data set object,
            # creating a new data set object if necessary.
            if (root_match := re.match(feature_dir_regex, root)):
                cls.logger.debug(f"Feature directory: {root}")
                cls.logger.debug(f"data set name: {root_match.group(1)}")
                cls.logger.debug(f"feature is: {root_match.group(2)}", )
                data_set_name = root_match.group(1)
                feature_name = root_match.group(2)
                if (data_set := data_sets.get(data_set_name)) is None:
                    data_set = DataSetObject(root_match.group(1), Path(root).parent)
                    data_sets[data_set.name] = data_set
                files = [f for f in files if not f.startswith(".") and not f.endswith(".tmp")]
                data_set.add_feature_files(feature_name, files)
        
        return data_sets

    def find_overlapping_time_section(
        self,
        features: List[str], # List of features included in the calculation, typically a combination of input and output features
        id: str, # Subject id to process
        ) -> Tuple[int, int]:
        '''
        Find common time interval when there's data for all features
        '''
        max_start = None
        min_end = None
        for feature in features:
            data = self.get_feature_data(feature, id)
            time = data[:, 0]
            if max_start is None:
                max_start = time.min()
            else:
                max_start = max([max_start, time.min()])
            if min_end is None:
                min_end = time.max()
            else:
                min_end = min([min_end, time.max()])
        return (max_start, min_end)

# %% ../nbs/01_data_sets.ipynb 13
def psg_to_sleep_wake(psg: pl.DataFrame) -> np.ndarray:
    """
    * map all positive classes to 1 (sleep)
    * retain all 0 (wake) and -1 (mask) classes
    """
    return np.where(psg[:, 1] > 0, 1, psg[:, 1])

def to_WLDM(x: float, N4: bool=True) -> int:
    """
    Map sleep stages to wake, light, deep, and REM sleep.
    Retain masked values. If N4 stage is not present,
    PSG=4 is mapped to REM. Otherwise it is mapped to deep sleep.
    """
    if x < 0:
        return -1
    if x == 0:
        return 0
    if x < 3:
        return 1
    rem_value = 5 if N4 else 4
    if x < rem_value:
        return 2
    return 3

vec_to_WLDM = np.vectorize(to_WLDM)

def psg_to_WLDM(psg: pl.DataFrame, N4: bool = True) -> np.ndarray:
    """
    * map all positive classes as follows:
    If N4 is True:
        - 1, 2 => 1 (light sleep)
        - 3, 4 => 2 (deep sleep)
        - 5 => 3 (REM)
    If N4 is False:
        - 1, 2 => 1 (light sleep)
        - 3 => 2 (deep sleep)
        - 5 => 3 (REM)
    * retain all 0 (wake) and -1 (mask) classes
    """
    return vec_to_WLDM(psg[:, 1].to_numpy(), N4)

# %% ../nbs/01_data_sets.ipynb 16
class ModelOutputType(Enum):
    SLEEP_WAKE = auto()
    WAKE_LIGHT_DEEP_REM = auto()

class PSGType(Enum):
    NO_N4 = auto()
    HAS_N4 = auto()

class ModelInput:
    def __init__(self,
                 input_features: List[str] | str,
                 input_sampling_hz: int | float, # Sampling rate of the input data (1/s)
                 ):
        # input_features
        if isinstance(input_features, str):
            input_features = [input_features]
        self.input_features = input_features
        # input_sampling_hz
        if not isinstance(input_sampling_hz, (int, float)):
            raise ValueError("input_sampling_hz must be an int or a float")
        else:
            if input_sampling_hz <= 0:
                raise ValueError("input_sampling_hz must be greater than 0")
        self.input_sampling_hz = float(input_sampling_hz)

class ModelInput1D(ModelInput):
    def __init__(self,
                 input_features: List[str] | str,
                 input_sampling_hz: int | float, # Sampling rate of the input data (1/s)
                 input_window_time: int | float, # Window size (in seconds) for the input data. Window will be centered around the time point for which the model is making a prediction
                 ):
        super().__init__(input_features, input_sampling_hz)
        # input_window_time
        if not isinstance(input_window_time, (int, float)):
            raise ValueError("input_window_time must be an int or a float")
        else:
            if input_window_time <= 0:
                raise ValueError("input_window_time must be greater than 0")

        self.input_window_time = float(input_window_time)
        # Number of samples for the input window of a single feature
        self.input_window_samples = int(self.input_window_time * self.input_sampling_hz)
        ## force it to be odd to have perfectly centered window
        if self.input_window_samples % 2 == 0:
            self.input_window_samples += 1
        # Dimension of the input data for the model
        self.model_input_dimension = int(len(input_features) * self. input_window_samples)

class ModelInputSpectrogram(ModelInput):
    def __init__(self,
                 input_features: List[str] | str,
                 input_sampling_hz: int | float, # Sampling rate of the input data (1/s)
                 spectrogram_preprocessing_config: Dict=MO_PREPROCESSING_CONFIG, # Steps in the preprocessing pipeline for getting a spectrogram from acceleration
                 ):
        super().__init__(input_features, input_sampling_hz)
        self.input_sampling_hz = float(input_sampling_hz)
        self.spectrogram_preprocessing_config = spectrogram_preprocessing_config

# %% ../nbs/01_data_sets.ipynb 17
def get_sample_weights(y: np.ndarray) -> np.ndarray:
     """
     Calculate sample weights based on the distribution of classes in the data.
     Doesn't count masked values (-1) in the class distribution.
     """
     # Filter out -1 values
     valid_y = y[y != -1]
     # Calculate class counts for valid labels only
     class_counts = np.bincount(valid_y)
     class_weights = np.where(class_counts > 0, class_counts.sum() / class_counts, 0)
     # Map valid class weights to corresponding samples in y
     sample_weights = np.zeros_like(y, dtype=float)
     for class_index, weight in enumerate(class_weights):
          sample_weights[y == class_index] = weight
     # Masked values (-1) in y will have a weight of 0
     return sample_weights


def mask_psg_from_accel(psg: np.ndarray, accel: np.ndarray, 
                        psg_epoch: int = 30,
                        accel_sample_rate: float | None = None,
                        min_epoch_fraction_covered: float = 0.5
                        ) -> np.ndarray:

    acc_last_index = 0
    acc_next_index = acc_last_index
    acc_last_time = accel[acc_last_index, 0]
    acc_next_time = acc_last_time

    # at least this fraction of 1 epoch must be covered
    # both in terms of time (no gap longer than 0.5 epochs)
    # and in terms of expected number of samples in that time.
    min_epoch_covered = min_epoch_fraction_covered * psg_epoch
    if accel_sample_rate is None:
        # median sample step size, if none provided
        # median to not take into account gaps!
        accel_sample_rate = np.median(np.diff(accel[:, 0]))
    min_samples_per = min_epoch_covered / accel_sample_rate

    psg_gap_indices = []

    for (psg_index, psg_sample) in enumerate(psg):
        epoch_ends = psg_sample[0] + psg_epoch

        # find the last timestamp inside the epoch
        while (acc_next_time <= epoch_ends and acc_next_index < len(accel)):
            acc_next_time = accel[acc_next_index, 0]
            acc_next_index += 1
        
        # 1. check for lots of missing time
        # 2. check for very low sampling rate
        if ((acc_next_time - acc_last_time) < min_epoch_covered) \
            or (acc_next_index - acc_last_index < min_samples_per):
            psg_gap_indices.append(psg_index)
        
        # set up for next iteration
        acc_last_time = acc_next_time
        acc_last_index = acc_next_index
    
    psg[np.array(psg_gap_indices), 1] = -1

    return psg


def apply_gausian_filter(df: pl.DataFrame, sigma: float = 1.0, overwrite: bool = False) -> pl.DataFrame:
    data_columns = df.columns[1:]  # Adjust this to match your data column indices
    # Apply Gaussian smoothing to each data column
    for col in data_columns:
        new_col_name = f"{col}_smoothed" if not overwrite else col
        df = df.with_columns(
            pl.Series(gaussian_filter1d(df[col].to_numpy(), sigma)).alias(new_col_name)
        )
    return df


def fill_gaps_in_accelerometer_data(acc: pl.DataFrame, smooth: bool = False, final_sampling_rate_hz: int | None = None) -> np.ndarray:
    # median sampling rate (to account for missing data)
    sampling_period_s = acc[acc.columns[0]].diff().median() # 1 / sampling_rate_hz
    
    # Step 0: Save the original 'timestamp' column as 'timestamp_raw'
    acc_resampled = acc.with_columns(acc[acc.columns[0]].alias('timestamp'))

    #TODO: Check non int sampling rates
    # if isinstance(final_sampling_rate_hz, int):
    final_rate_sec = 1 / final_sampling_rate_hz
    print(f"resampling to {final_sampling_rate_hz}Hz ({final_rate_sec:0.5f}s) from {int(1/sampling_period_s)} Hz ({sampling_period_s:0.5f}s)")
    # make a new data frame with the new timestamps
    # do this using linear interpolation

    median_time = acc_resampled['timestamp'].to_numpy()
    final_timestamps = np.arange(median_time.min(), median_time.max() + final_rate_sec, final_rate_sec)
    median_data = acc_resampled[:, 1:4].to_numpy()
    new_data = np.zeros((final_timestamps.shape[0], median_data.shape[1]))
    for i in range(median_data.shape[1]):
        new_data[:, i] = np.interp(final_timestamps, median_time, median_data[:, i])
    acc_resampled = pl.DataFrame({
        'timestamp': final_timestamps, 
        **{
            acc_resampled.columns[i+1]: new_data[:, i] 
            for i in range(new_data.shape[1])
        }})

    if smooth:
        acc_resampled = apply_gausian_filter(acc_resampled, overwrite=True)

    return acc_resampled

# %% ../nbs/01_data_sets.ipynb 18
class DataProcessor:
    def __init__(self,
                 data_set: DataSetObject,
                 model_input: ModelInput,
                 output_feature: str='psg',
                 output_type: ModelOutputType=ModelOutputType.WAKE_LIGHT_DEEP_REM,
                 psg_type: PSGType=PSGType.NO_N4,
                 ):
        self.data_set = data_set
        self.input_features = model_input.input_features
        self.output_feature = output_feature
        self.output_type = output_type
        self.psg_type = psg_type
        self.model_input = model_input

        if self.is_1D:
            self.input_window_time = model_input.input_window_time
            self.input_sampling_hz = model_input.input_sampling_hz
            self.input_window_samples = model_input.input_window_samples
            self.model_input_dimension = model_input.model_input_dimension
        elif self.is_spectrogram:
            self.input_sampling_hz = model_input.input_sampling_hz
            self.spectrogram_preprocessing_config = model_input.spectrogram_preprocessing_config

    @property
    def is_1D(self):
        return isinstance(self.model_input, ModelInput1D)

    @property
    def is_spectrogram(self):
        return isinstance(self.model_input, ModelInputSpectrogram)

    def get_labels(self, id: str, start: int, end: int,
                   output_feature: str) -> pl.DataFrame | None:
        data = self.data_set.get_feature_data(output_feature, id)
        data = data.filter(data[:, 0] >= start)
        data = data.filter(data[:, 0] <= end)

        # Mask PSG data based on accelerometer data if present
        if "accelerometer" in self.input_features:
            accelerometer = self.data_set.get_feature_data('accelerometer', id)
            data = mask_psg_from_accel(data, accelerometer, 
                                       accel_sample_rate=self.input_sampling_hz)

        if self.output_feature == 'psg':
            if self.output_type == ModelOutputType.SLEEP_WAKE:
                y = psg_to_sleep_wake(data)
            elif self.output_type == ModelOutputType.WAKE_LIGHT_DEEP_REM:
                N4 = self.psg_type == PSGType.HAS_N4
                y = psg_to_WLDM(data, N4)
            else:
                raise ValueError(f"Output type {self.output_type} not supported")
        else:
            raise ValueError(f"Output feature {output_feature} not supported")

        labels = pl.DataFrame({
            'time': data[:, 0],
            'label': y,
        })

        return labels

    def get_1D_X_for_feature(self, interpolation_timestamps: np.ndarray, 
                             epoch_times: np.ndarray, feature_times: np.ndarray, 
                             feature_values: np.ndarray) -> np.ndarray:
            interpolation = np.interp(interpolation_timestamps, feature_times, feature_values)
            X_feature = []
            for t in epoch_times:
                t_idx = np.argmin(np.abs(interpolation_timestamps - t))
                # Window centered around t with half `window_samples` on each side
                window_idx_start = t_idx - self.input_window_samples // 2
                window_idx_end = t_idx + self.input_window_samples // 2 + 1
                window_data = interpolation[window_idx_start:window_idx_end]
                # reshape into (1, window_size)
                window_data = window_data.reshape(1, -1)
                X_feature.append(window_data)
            # create a numpy array of shape (n_samples, window_size)
            X_feature = np.vstack(X_feature)
            return X_feature

    def get_1D_X_y(self, id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        # Find overlapping time section
        all_features = self.input_features + [self.output_feature]
        max_start, min_end = self.data_set.find_overlapping_time_section(all_features, id)
        # Get labels
        labels = self.get_labels(id, max_start, min_end, self.output_feature)
        label_times = labels[:, 0]
        epoch_start = label_times.min() + self.input_window_time / 2.0
        epoch_end = label_times.max() - self.input_window_time / 2.0
        filtered_labels = labels.filter(labels[:, 0] >= epoch_start)
        filtered_labels = filtered_labels.filter(filtered_labels[:, 0] <= epoch_end)
        epoch_times = filtered_labels[:, 0]
        # Get input data
        interpolation_timestamps = np.arange(max_start, 
                                             min_end + 1.0/self.input_sampling_hz,
                                             1.0/self.input_sampling_hz,)
        # Interpolate all data to the same time points
        interpolated_features = []
        for feature in self.input_features:
            data = self.data_set.get_feature_data(feature, id)
            feature_times = data[:, 0]
            if feature == 'accelerometer':
                # Handle accelerometer data
                for i in range(1, 4):
                    feature_values = data[:, i]
                    X_feature = self.get_1D_X_for_feature(interpolation_timestamps, 
                                                          epoch_times, feature_times, 
                                                          feature_values)
                    interpolated_features.append(X_feature)
            else:
                feature_values = data[:, 1]
                X_feature = self.get_1D_X_for_feature(interpolation_timestamps, 
                                                      epoch_times, feature_times, 
                                                      feature_values)
                interpolated_features.append(X_feature)
        # Concatenate input features alongside the first dimension
        X = np.concatenate(interpolated_features, axis=1)
        y = filtered_labels[:, 1].to_numpy()
        return X, y
    
    def accelerometer_to_spectrogram(self, accelerometer: pl.DataFrame) -> np.ndarray:
        """
        Implementation by Mads Olsen at https://github.com/MADSOLSEN/SleepStagePrediction
        with minor modifications.
        """
        if isinstance(accelerometer, pl.DataFrame):
            acc = accelerometer.to_numpy()
        else:
            raise ValueError("accelerometer must be a polars DataFrame")

        x_ = acc[:, 1]
        y_ = acc[:, 2]
        z_ = acc[:, 3]

        for step in self.spectrogram_preprocessing_config["preprocessing"]:
            fn = eval(step["type"])  # convert string version to function in environment
            fn_args = partial(
                fn, **step["args"]
            )  # fill in the args given, which must be everything besides numerical input

            # apply
            x_ = fn_args(x_)
            y_ = fn_args(y_)
            z_ = fn_args(z_)

        spec = x_ + y_ + z_
        spec /= 3.0

        return spec

    def mirror_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        # We will copy the spectrogram to both channels, flipping it on channel 1
        input_shape = (1, *MO_UNET_CONFIG['input_shape'])
        inputs_len = input_shape[1]

        mirrored = np.zeros(shape=input_shape, dtype=np.float32)
        # We must do some careful work with indices to not overflow arrays
        spec = spectrogram[:inputs_len].astype(np.float32) # protect agains spec.len > input_shape

        #! careful, order matters here. We first trim spec to make sure it'll fit into inputs,
        # then compute the new length which we KNOW is <= inputs_len
        spec_len = spec.shape[0]
        # THEN we assign only as much inputs as spec covers
        mirrored[0, : spec_len, :, 0] = spec # protect agains spec_len < input_shape
        mirrored[0, : spec_len, :, 1] = spec[:, ::-1]

        return mirrored

    def get_spectrogram_X_y(self, id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        # Find overlapping time section
        all_features = self.input_features + [self.output_feature]
        max_start, min_end = self.data_set.find_overlapping_time_section(all_features, id)

        if self.input_features != ['accelerometer']:
            raise ValueError("Spectrogram input only supported for accelerometer data")

        accelerometer = self.data_set.get_feature_data('accelerometer', id)
        # Use only the overlapping time section
        accelerometer = accelerometer.filter(accelerometer[:, 0] >= max_start)
        accelerometer = accelerometer.filter(accelerometer[:, 0] <= min_end)
        # Fill gaps in accelerometer data
        accelerometer = fill_gaps_in_accelerometer_data(accelerometer, smooth=False, 
                                                        final_sampling_rate_hz=self.input_sampling_hz)
        # Get spectrogram (mirrored)
        spectrogram = self.accelerometer_to_spectrogram(accelerometer)
        X = self.mirror_spectrogram(spectrogram)

        # Get labels
        labels = self.get_labels(id, max_start, min_end, self.output_feature)
        y = labels[:, 1].to_numpy()
        # Match labels to model output
        if len(y) < N_OUT:
            y = np.pad(y, (0, N_OUT - len(y)), constant_values=-1)
        elif len(y) > N_OUT:
            y = y[:N_OUT]

        return X, y 

    def preprocess_data_for_subject(self, id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        if self.is_1D:
            return self.get_1D_X_y(id)
        elif self.is_spectrogram:
            return self.get_spectrogram_X_y(id)
        else:
            raise ValueError("ModelInput type not supported")
