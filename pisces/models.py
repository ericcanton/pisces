# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_models.ipynb.

# %% auto 0
__all__ = ['SleepWakeClassifier', 'train_pipeline', 'LinearModel', 'SGDLinearClassifier', 'RandomForest', 'MOResUNetPretrained',
           'SplitMaker', 'LeaveOneOutSplitter', 'run_split', 'run_splits']

# %% ../nbs/02_models.ipynb 4
import sys
import abc
import keras
import warnings
import numpy as np
import polars as pl
from enum import Enum
from tqdm import tqdm
import multiprocessing
from io import StringIO
from typing import Type
from itertools import repeat
from scipy.special import softmax
from fastcore.basics import patch_to
from typing import Dict, List, Tuple
from sklearn.pipeline import Pipeline
from .mads_olsen_support import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from .mads_olsen_support import load_saved_keras
from .data_sets import ModelInput1D, ModelInputSpectrogram, ModelOutputType, DataProcessor

# %% ../nbs/02_models.ipynb 6
class SleepWakeClassifier:
    """ Abstract class for sleep/wake classifiers. 
    """
    def __init__(self, model=None, data_processor=None,
                 scaler_pipeline_name: str='scaler', 
                 model_pipeline_name: str='model'):
        self.model = model
        self.scaler_pipeline_name = scaler_pipeline_name
        self.model_pipeline_name = model_pipeline_name
        self.pipeline = Pipeline([(scaler_pipeline_name, StandardScaler()), 
                                  (model_pipeline_name, self.model)])
        self.data_processor = data_processor

    def _input_preprocessing(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.named_steps[self.scaler_pipeline_name].transform(X)

    def predict(self, sample_X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """
        Assumes data is already preprocessed using `get_needed_X_y`
        """
        return self.pipeline.predict(self._input_preprocessing(sample_X))

    def predict_probabilities(self, sample_X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """
        Assumes data is already preprocessed using `get_needed_X_y`
        """
        model = self.pipeline.named_steps[self.model_pipeline_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(self._input_preprocessing(sample_X))
        elif hasattr(model, 'decision_function'):
            warnings.warn("Model does not have `predict_proba`. Using `decision_function` instead.")
            binary_decision = model.decision_function(self._input_preprocessing(sample_X))
            prediction = binary_decision > 0
            return np.vstack([1 - prediction, prediction]).T

        else:
            raise ValueError("Model must have either `predict_proba` or `decision_function`")

# %% ../nbs/02_models.ipynb 7
def train_pipeline(classifier: SleepWakeClassifier,
                   examples_X: List[np.ndarray]=[],
                   examples_y: List[np.ndarray]=[],
                   pairs_Xy: List[Tuple[np.ndarray, np.ndarray]]=[],
                   **train_kwargs) -> List[float]:
    """
    Assumes data is already preprocessed using `get_needed_X_y` 
    and ready to be passed to the classifier.

    Returns the loss history of the model.

    Parameters
    ----------
    classifier : SleepWakeClassifier
        The classifier object.
    examples_X : List[np.ndarray]
        List of input examples. If non-empty, then `examples_y` must also be provided and must have the same length.
    examples_y : List[np.ndarray]
        List of target labels. If non-empty, then `examples_X` must also be provided and must have the same length.
    pairs_Xy : List[Tuple[np.ndarray, np.ndarray]]
        List of input-target pairs. If non-empty, then `examples_X` and `examples_y` must not be provided.

    Returns
    -------
    List[float]
        The loss history of the model.
    """
    if pairs_Xy:
        assert (not examples_X) and (not examples_y)
        examples_X = [pair[0] for pair in pairs_Xy]
        examples_y = [pair[1] for pair in pairs_Xy]
    elif examples_X and not examples_y: 
        raise ValueError("Provided examples_X but not examples_y")
    elif examples_y and not examples_X:
        raise ValueError("Provided examples_y but not examples_X")
    else:
        # we know that examples_X and examples_y are both truthy, hence non-empty lists
        assert (len(examples_X) == len(examples_y))

    # for j in range(len(examples_y)):
    #     if len(examples_y[j].shape) == 1 or examples_y[j].shape[1] == 1:
    #         print(f"reshaping {examples_y[j].shape}")
    #         examples_y[j] = examples_y[j].reshape(-1, 1)
    #         print("now shaped to", examples_y[j].shape)
    examples_y = [
        y.reshape(-1, )
        for y in examples_y]
        #  if len(y.shape) == 1 or y.shape[1] == 1 else y ]

    Xs = np.concatenate(examples_X, axis=0)
    ys = np.concatenate(examples_y, axis=0)
    print(f"Training on {len(Xs)} examples")

    selector = ys >= 0

    # adds "model__sample_weight" to train_kwargs, but if it already exists, it will not overwrite it
    # the order of the dictionaries in | important, as the rightmost dictionary will overwrite the leftmost
    train_kwargs = {classifier.model_pipeline_name + '__sample_weight': selector} \
        | train_kwargs

    loss_list = []
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    # multiply ys by selector, to zero-out the "-1" masked values but leave the others unchanged (where selector == 1)
    classifier.pipeline.fit(Xs, ys * selector, **train_kwargs) # Fit the model
    print("Done fitting")
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    print(loss_history)
    # Get loss
    try:
        for line in loss_history.split('\n'):
            if(len(line.split("loss: ")) == 1):
                continue
            loss_list.append(float(line.split("loss: ")[-1]))
    except:
        warnings.warn("Failed to fetch loss history. Returning empty list.")

    return loss_list

# %% ../nbs/02_models.ipynb 8
@patch_to(SleepWakeClassifier)
def train(self, examples_X: List[np.ndarray]=[], 
          examples_y: List[np.ndarray]=[], 
          pairs_Xy: List[Tuple[np.ndarray, np.ndarray]]=[],
          **training_kwargs
          ):
    """
    Assumes data is already preprocessed using `get_needed_X_y` 
    and ready to be passed to the model.

    Returns the loss history of the model.
    """
    loss_list = train_pipeline(self, examples_X, examples_y, pairs_Xy, **training_kwargs)
    return loss_list

# %% ../nbs/02_models.ipynb 10
class LinearModel(Enum):
    """Defines the loss used in sklearn's SGDClassifier which defines the linear model used for classification."""
    LOGISTIC_REGRESSION = 'log_loss'
    PERCEPTRON = 'perceptron'
    SVM = 'hinge'

# %% ../nbs/02_models.ipynb 11
class SGDLinearClassifier(SleepWakeClassifier):
    """Uses Sk-Learn's `SGDCLassifier` to train a model. Possible models are logistic regression, perceptron, and SVM.
    The SGD aspect allows for online learning, or custom training regimes through the `partial_fit` method.
    The model is trained with a balanced class weight, and uses L1 regularization. The input data is scaled with a `StandardScaler` before being passed to the model.
    """
    def __init__(self, 
                 data_processor: DataProcessor | None = None, 
                 linear_model: LinearModel=LinearModel.LOGISTIC_REGRESSION,
                 **kwargs):
        if data_processor is not None:
            if not isinstance(data_processor.model_input, ModelInput1D):
                raise ValueError("Model input must be set to 1D on the data processor")
            if not data_processor.output_type == ModelOutputType.SLEEP_WAKE:
                raise ValueError("Model output must be set to SleepWake on the data processor")
        super().__init__(
            model=SGDClassifier(loss=linear_model.value, **kwargs),
            data_processor=data_processor
        )

    def get_needed_X_y(self, id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        return self.data_processor.get_1D_X_y(id)

# %% ../nbs/02_models.ipynb 13
class RandomForest(SleepWakeClassifier):
    """Interface for sklearn's RandomForestClassifier"""
    def __init__(self,
                 data_processor: DataProcessor | None = None,
                 class_weight: str = 'balanced',
                 **kwargs):
        if data_processor is not None:
            if not isinstance(data_processor.model_input, ModelInput1D):
                raise ValueError("Model input must be set to 1D on the data processor")
        super().__init__(
            model=RandomForestClassifier(class_weight=class_weight, **kwargs),
            data_processor=data_processor
        )

    def get_needed_X_y(self, id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        return self.data_processor.get_1D_X_y(id)

# %% ../nbs/02_models.ipynb 15
class MOResUNetPretrained(SleepWakeClassifier):

    def __init__(
        self,
        data_processor: DataProcessor | None = None,
        model: keras.Model | None = None,
        lazy_model_loading: bool = True,
        initial_lr: float = 1e-5,
        validation_split: float = 0.1,
        epochs: int = 10,
        batch_size: int = 1,
        **kwargs) -> None:
        """
        Initialize the MOResUNetPretrained classifier.

        Args:
            data_processor (DataProcessor, optional): The data processor to use.
            model (keras.Model, optional): The TensorFlow model to use. Defaults to None, in which case the model is loaded from disk.
        """
        if data_processor is not None:
            if not isinstance(data_processor.model_input, ModelInputSpectrogram):
                raise ValueError("Model input must be set to Spectrogram on the data processor")
        self.initial_lr = initial_lr

        self.model = model
        if model is None and not lazy_model_loading:
            self.load_model(force=True)

        super().__init__(
            model=self.model,
            data_processor=data_processor,
        )

        
        # set up training params using the named step format of a pipeline.fit **kwargs
        self.training_params = {
            f'{self.model_pipeline_name}__validation_split': validation_split,
            f'{self.model_pipeline_name}__epochs': epochs,
            f'{self.model_pipeline_name}__batch_size': batch_size
        }

    
    def load_model(self, force: bool=True) -> None:
        if self.model is not None and not force:
            return
        self.model = load_saved_keras()
        self.pipeline = Pipeline([
            (self.model_pipeline_name, self.model)
        ])
        if self.model is not None:
            self.model.compile(
                optimizer=keras.optimizers.RMSprop(learning_rate=self.initial_lr),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
                weighted_metrics=[])
        else:
            raise ValueError("Model could not be loaded.")



    def prepare_set_for_training(self, 
                                 ids: List[str],
                                 max_workers: int | None = None 
                                 ) -> List[Tuple[np.ndarray, np.ndarray] | None]:
        """
        Prepare the data set for training.

        Args:
            ids (List[str], optional): The IDs to prepare. Defaults to None.
            max_workers (int, optional): The number of workers to use for parallel processing. Defaults to None, which uses all available cores. Setting to a negative number leaves that many cores unused. For example, if my machine has 4 cores and I set max_workers to -1, then 3 = 4 - 1 cores will be used; if max_workers=-3 then 1 = 4 - 3 cores are used.

        Returns:
            List[Tuple[np.ndarray, np.ndarray] | None]: A list of tuples, where each tuple is the result of `get_needed_X_y` for a given ID. An empty list indicates an error occurred during processing.
        """
        results = []
        
        # Get the number of available CPU cores
        num_cores = multiprocessing.cpu_count()
        workers_to_use = max_workers if max_workers is not None else num_cores
        if (workers_to_use > num_cores):
            warnings.warn(f"Attempting to use {max_workers} but only have {num_cores}. Running with {num_cores} workers.")
            workers_to_use = num_cores
        if workers_to_use <= 0:
            workers_to_use = num_cores + max_workers
        if workers_to_use < 1:
            # do this check second, NOT with elif, to verify we're still in a valid state
            raise ValueError(f"With `max_workers` == {max_workers}, we end up with f{max_workers + num_cores} ({max_workers} + {num_cores}) which is less than 1. This is an error.")

        print(f"Using {workers_to_use} of {num_cores} cores ({int(100 * workers_to_use / num_cores)}%) for parallel preprocessing.")
        print(f"This can cause memory or heat issues if  is too high; if you run into problems, call prepare_set_for_training() again with max_workers = -1, going more negative if needed. (See the docstring for more info.)")
        # Create a pool of workers
        with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.get_needed_X_y,
                        ids,
                        repeat(self.data_processor),
                    ), total=len(ids), desc="Preparing data..."
                ))

        return results

    def get_needed_X_y(self, id: str, data_processor: DataProcessor) -> Tuple[np.ndarray, np.ndarray] | None:
        return data_processor.get_spectrogram_X_y(id)

    def predict(self, sample_X: np.ndarray | pl.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_probabilities(sample_X), axis=1)

    def predict_probabilities(self, sample_X: np.ndarray | pl.DataFrame) -> np.ndarray:
        if isinstance(sample_X, pl.DataFrame):
            sample_X = sample_X.to_numpy()
        return softmax(self._evaluate_tf_model(sample_X)[0], axis=1)

    def _evaluate_tf_model(self, inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.astype(np.float32)
        preds = self.model.predict(inputs)
        return preds

    # def evaluate_data_set(self, 
    #                       exclude: List[str] = [], 
    #                       max_workers: int = None) -> Tuple[Dict[str, dict], list]:
    #     data_set = self.data_processor.data_set
    #     filtered_ids = [id for id in data_set.ids if id not in exclude]
    #     # Prepare the data
    #     print("Preprocessing data...")
    #     mo_preprocessed_data = [
    #         (d, i) 
    #         for (d, i) in zip(
    #             self.prepare_set_for_training(filtered_ids, max_workers=max_workers),
    #             filtered_ids) 
    #         if d is not None
    #     ]

    #     print("Evaluating data set...")
    #     evaluations: Dict[str, dict] = {}
    #     for _, ((X, y_true), id) in tqdm(enumerate(mo_preprocessed_data)):
    #         y_prob = self.predict_probabilities(X)
    #         m = keras.metrics.SparseCategoricalAccuracy()
    #         # Remove masked values
    #         selector = y_true >= 0
    #         y_true_filtered = y_true[selector]
    #         y_prob_filtered = y_prob[selector]
    #         # Calculate sample weights
    #         unique, counts = np.unique(y_true_filtered, return_counts=True)
    #         class_weights = dict(zip(unique, counts))
    #         inv_class_weights = {k: 1.0 / v for k, v in class_weights.items()}
    #         min_weight = min(inv_class_weights.values())
    #         normalized_weights = {k: v / min_weight for k, v in inv_class_weights.items()}
    #         sample_weights = np.array([normalized_weights[class_id] for class_id in y_true_filtered])
    #         # Sparse categorical accuracy
    #         y_true_reshaped = y_true_filtered.reshape(-1, 1)
    #         m.update_state(y_true_reshaped, y_prob_filtered, sample_weight=sample_weights)
    #         accuracy = m.result().numpy()
    #         evaluations[id] = {
    #             'sparse_categorical_accuracy': accuracy,
    #         }

    #     return evaluations, mo_preprocessed_data

# %% ../nbs/02_models.ipynb 17
class SplitMaker:
    def split(self, ids: List[str]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError
    
class LeaveOneOutSplitter(SplitMaker):
    def split(self, ids: List[str]) -> Tuple[List[int], List[int]]:
        loo = LeaveOneOut()
        return loo.split(ids)


def run_split(train_indices, 
              preprocessed_data_set: List[Tuple[np.ndarray, np.ndarray]],
              swc: SleepWakeClassifier,
              epochs: int| None, 
              do_not_train: bool = False) -> Tuple[SleepWakeClassifier, List[float]]:
    if do_not_train:
        return swc, []
    training_pairs = [
        [preprocessed_data_set[i][0], preprocessed_data_set[i][1].reshape(1, -1)]
        for i in train_indices
        if preprocessed_data_set.get(i) is not None
    ]
    if isinstance(swc, MOResUNetPretrained):
        extra_params = {
            f'{swc.model_pipeline_name}__epochs': epochs,
            f'{swc.model_pipeline_name}__batch_size': 1,
            f'{swc.model_pipeline_name}__validation_split': 0.1
        }
        print(swc.pipeline.named_steps)
        result = swc.train(pairs_Xy=training_pairs, **extra_params)
    else:
        result = swc.train(pairs_Xy=training_pairs)

    return swc, result


def run_splits(split_maker: SplitMaker, 
               data_processor: DataProcessor, 
               swc_class: Type[SleepWakeClassifier], 
               epochs: int | None,
               exclude: List[str] = [],
               linear_model: LinearModel | None = None,
               ) -> Tuple[
                   List[SleepWakeClassifier], 
                   List[np.ndarray], 
                   List[List[List[int]]] 
                   ]:
    split_models: List[swc_class] = []
    test_indices = []
    split_results = []
    splits = []

    if swc_class == SGDLinearClassifier and not linear_model:
        raise ValueError("Must provide a linear model for SGDLinearClassifier")
    elif not swc_class == SGDLinearClassifier and linear_model:
        raise ValueError("Linear model provided but not using SGDLinearClassifier")

    swc = swc_class(data_processor, linear_model=linear_model, epochs=epochs)

    ids_to_split = [id for id in data_processor.data_set.ids if id not in exclude]
    tqdm_message_preprocess = f"Preparing data for {len(ids_to_split)} IDs"
    preprocessed_data = [(swc.get_needed_X_y(id), id) for id in tqdm(ids_to_split, desc=tqdm_message_preprocess)]

    tqdm_message_train = f"Training {len(ids_to_split)} splits"
    all_splits = split_maker.split(ids_to_split)
    for train_index, test_index in tqdm(all_splits, desc=tqdm_message_train, total=len(ids_to_split)):
        if preprocessed_data[test_index[0]][0] is None:
            continue
        if swc_class == SGDLinearClassifier:
            swc = swc_class(data_processor, linear_model, 
                            epochs=epochs)
        elif swc_class == RandomForest:
            swc = swc_class(data_processor)
        else:
            swc = swc_class(data_processor, epochs=epochs)

        model, result = run_split(train_indices=train_index, 
                                  preprocessed_data_set=preprocessed_data, 
                                  swc=swc,
                                  epochs=epochs)
        split_models.append(model)
        split_results.append(result)
        test_indices.append(test_index[0])
        splits.append([train_index, test_index])
    
    return split_models, split_results, preprocessed_data, splits
