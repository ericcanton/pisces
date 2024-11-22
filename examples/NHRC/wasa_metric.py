import jax
import jax.numpy as jnp
from keras.metrics import Metric


def find_best_threshold_jnp(y_true, y_pred, weights, target_accuracy, tol=1e-4):
    """
    Perform a binary search to find the best threshold for the given target accuracy.
    """

    def compute_accuracy(threshold):
        y_pred_bin = (y_pred >= threshold).astype(jnp.float32)
        correct_predictions = jnp.sum(y_pred_bin * true_positive_mask)
        return correct_predictions / (n_true_positives + 1e-10)

    # Mask for true positives
    true_positive_mask = (y_true == 1) & (weights > 0)
    n_true_positives = jnp.sum(true_positive_mask)

    # Binary search variables
    threshold_min = 0.0
    threshold_max = 1.0

    def cond_fn(state):
        threshold_min, threshold_max, _ = state
        return (threshold_max - threshold_min) > tol

    def body_fn(state):
        threshold_min, threshold_max, _ = state
        threshold = (threshold_min + threshold_max) / 2.0
        accuracy = compute_accuracy(threshold)

        threshold_min = jax.lax.cond(
            accuracy >= target_accuracy,
            lambda _: threshold,
            lambda _: threshold_min,
            operand=None,
        )
        threshold_max = jax.lax.cond(
            accuracy < target_accuracy,
            lambda _: threshold,
            lambda _: threshold_max,
            operand=None,
        )

        return threshold_min, threshold_max, threshold

    # Initial state: (threshold_min, threshold_max, best_threshold)
    initial_state = (threshold_min, threshold_max, 0.5)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    _, _, best_threshold = final_state

    return best_threshold


class WASAMetric(Metric):
    """
    Custom Keras metric to compute the optimal threshold for achieving a target sleep accuracy.
    """

    def __init__(self, sleep_accuracy: float = 0.95, class_id: int = None, **kwargs):
        super(WASAMetric, self).__init__(name=f"wasa_{int(100 * sleep_accuracy)}", **kwargs)
        self.sleep_accuracy = sleep_accuracy
        self.class_id = class_id

        # Metric state variables
        self.true_sleep = self.add_weight(name="true_sleep", initializer="zeros")
        self.false_wake = self.add_weight(name="false_wake", initializer="zeros")
        self.true_wake = self.add_weight(name="true_wake", initializer="zeros")
        self.false_sleep = self.add_weight(name="false_sleep", initializer="zeros")

    def compute_true_false_pos_neg(self, y_true, y_pred, sample_weight, threshold):
        """
        Compute true positives, false negatives, true negatives, and false positives.
        """
        y_pred_bin = (y_pred >= threshold).astype(jnp.float32)

        tp = (y_true == 1) & (y_pred_bin == 1)
        fn = (y_true == 1) & (y_pred_bin == 0)
        tn = (y_true == 0) & (y_pred_bin == 0)
        fp = (y_true == 0) & (y_pred_bin == 1)

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
        y_true = jnp.reshape(y_true, [-1])
        y_pred = jnp.reshape(1 - y_pred[..., 0], [-1])  # Convert probability to binary class score
        sample_weight = jnp.reshape(sample_weight, [-1]) if sample_weight is not None else None

        # Find the optimal threshold for the target accuracy
        threshold = find_best_threshold_jnp(y_true, y_pred, sample_weight, self.sleep_accuracy)

        # Compute true/false positives/negatives
        tp, fn, tn, fp = self.compute_true_false_pos_neg(y_true, y_pred, sample_weight, threshold)

        # Sum up the metrics and update the state
        self.true_sleep.assign_add(jnp.sum(tp))
        self.false_wake.assign_add(jnp.sum(fn))
        self.true_wake.assign_add(jnp.sum(tn))
        self.false_sleep.assign_add(jnp.sum(fp))

    def result(self):
        """
        Compute and return the sensitivity of the metric.
        """
        sensitivity = self.true_sleep / (self.true_sleep + self.false_wake + 1e-10)
        return sensitivity

    def reset_states(self):
        """
        Reset the metric state variables.
        """
        for variable in self.variables:
            variable.assign(0)
