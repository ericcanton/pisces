import tensorflow as tf
from keras.metrics import Metric, SpecificityAtSensitivity
from keras.layers import Reshape
import keras.backend as K


def find_best_threshold(y_true, y_pred, weights, target_accuracy, tol=1e-4):
    """
    Perform a binary search to find the best threshold for the given target accuracy.
    """

    # Mask for true positives
    true_positive_mask = tf.cast((y_true == 1) & (weights > 0), tf.float32)
    n_true_positives = tf.reduce_sum(true_positive_mask)

    def compute_accuracy(threshold):
        y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
        correct_predictions = tf.reduce_sum(y_pred_bin * true_positive_mask)
        return correct_predictions / (n_true_positives + K.epsilon())

    # Binary search variables
    threshold_min = tf.constant(0.0)
    threshold_max = tf.constant(1.0)

    def cond(threshold_min, threshold_max, _):
        return tf.greater(threshold_max - threshold_min, tol)

    def body(threshold_min, threshold_max, _):
        threshold = (threshold_min + threshold_max) / 2.0
        accuracy = compute_accuracy(threshold)

        threshold_min = tf.where(accuracy >= target_accuracy, threshold, threshold_min)
        threshold_max = tf.where(accuracy < target_accuracy, threshold, threshold_max)

        return threshold_min, threshold_max, threshold

    # Initial state
    threshold_min, threshold_max, best_threshold = tf.while_loop(
        cond, body, [threshold_min, threshold_max, 0.5]
    )

    return best_threshold


class WASAMetric(Metric):
    """
    Custom Keras metric to compute the optimal threshold for achieving a target sleep accuracy.
    """

    def __init__(self, sleep_accuracy: float = 0.95, class_id: int = None, **kwargs):
        super().__init__(name=f"wasa_{int(100 * sleep_accuracy)}", **kwargs)
        self.sleep_accuracy = sleep_accuracy
        self.class_id = class_id
        self.keras_metric = SpecificityAtSensitivity(sensitivity=0.95, class_id=class_id)

        # # Metric state variables
        # self.true_sleep = self.add_weight(name="true_sleep", initializer="zeros")
        # self.false_wake = self.add_weight(name="false_wake", initializer="zeros")
        # self.true_wake = self.add_weight(name="true_wake", initializer="zeros")
        # self.false_sleep = self.add_weight(name="false_sleep", initializer="zeros")

    def compute_true_false_pos_neg(self, y_true, y_pred, sample_weight, threshold):
        """
        Compute true positives, false negatives, true negatives, and false positives.
        """
        y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)

        tp = tf.cast((y_true == 1) & (y_pred_bin == 1), tf.float32)
        fn = tf.cast((y_true == 1) & (y_pred_bin == 0), tf.float32)
        tn = tf.cast((y_true == 0) & (y_pred_bin == 0), tf.float32)
        fp = tf.cast((y_true == 0) & (y_pred_bin == 1), tf.float32)

        if sample_weight is not None:
            tp = tp * sample_weight
            fn = fn * sample_weight
            tn = tn * sample_weight
            fp = fp * sample_weight

        return tp, fn, tn, fp

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state based on current predictions.
        """
        # Flatten arrays
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(1 - y_pred[..., 0], [-1])  # Convert probability to binary class score
        sample_weight = (
            tf.reshape(sample_weight, [-1]) if sample_weight is not None else None
        )

        self.keras_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Compute and return the sensitivity of the metric.
        """
        # sensitivity = self.true_sleep / (self.true_sleep + self.false_wake + K.epsilon())
        # return sensitivity
        return self.keras_metric.result()

    def reset_states(self):
        """
        Reset the metric state variables.
        """
        self.keras_metric.reset_states()
