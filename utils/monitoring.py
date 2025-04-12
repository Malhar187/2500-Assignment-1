# utils/monitoring.py
from prometheus_client import start_http_server, Gauge, Counter
import threading
import time

class TrainingMonitor:
    def __init__(self, port=8000):
        self.port = port
        self._start_metrics_server()

        # General training metrics
        self.epoch = Gauge('training_epoch', 'Current training epoch')
        self.batch = Gauge('training_batch', 'Current batch number in epoch')
        self.train_loss = Gauge('training_loss', 'Training loss')
        self.val_loss = Gauge('validation_loss', 'Validation loss')
        self.val_accuracy = Gauge('validation_accuracy', 'Validation accuracy')
        
        # Resource usage (if extended)
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')  # Optional, if GPU available

    def _start_metrics_server(self):
        def run_server():
            start_http_server(self.port)
            while True:
                time.sleep(1)

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    def update_training_metrics(self, epoch=None, batch=None, train_loss=None):
        if epoch is not None:
            self.epoch.set(epoch)
        if batch is not None:
            self.batch.set(batch)
        if train_loss is not None:
            self.train_loss.set(train_loss)

    def update_validation_metrics(self, val_loss=None, val_accuracy=None):
        if val_loss is not None:
            self.val_loss.set(val_loss)
        if val_accuracy is not None:
            self.val_accuracy.set(val_accuracy)

# ------------------ REGRESSION MONITOR ------------------

class RegressionMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)

        # Regression-specific metrics
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])

    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        if mse is not None:
            self.mse.set(mse)
        if rmse is not None:
            self.rmse.set(rmse)
        if mae is not None:
            self.mae.set(mae)
        if r_squared is not None:
            self.r_squared.set(r_squared)

        if feature_importance is not None:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)

# ------------------ TREE-BASED MODEL MONITOR ------------------

class TreeModelMonitor(TrainingMonitor):
    def __init__(self, port=8003):
        super().__init__(port)

        self.tree_depth = Gauge('tree_max_depth', 'Maximum tree depth')
        self.tree_leaves = Gauge('tree_leaf_count', 'Number of leaf nodes')
        self.trees_count = Gauge('ensemble_tree_count', 'Number of trees in the ensemble')

        self.boost_round = Counter('boosting_rounds_total', 'Total boosting rounds completed')
        self.iteration_improvement = Gauge('iteration_improvement', 'Improvement in last iteration')

    def record_tree_metrics(self, depth=None, leaves=None, trees=None):
        if depth is not None:
            self.tree_depth.set(depth)
        if leaves is not None:
            self.tree_leaves.set(leaves)
        if trees is not None:
            self.trees_count.set(trees)

    def record_boost_round(self, improvement=None):
        self.boost_round.inc()
        if improvement is not None:
            self.iteration_improvement.set(improvement)