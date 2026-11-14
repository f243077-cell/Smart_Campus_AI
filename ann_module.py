"""
ann_module.py
-----------------------------------------------------------------------
Module 3.3 - ANN Priority Module
Predicts the urgency / priority of a request.

  * Perceptron : binary baseline / comparison model
                 (1 = urgent, 0 = not_urgent)
  * MLP        : final operational multiclass model 7 -> 4 -> 3 -> 4
                 (urgent, high, normal, low) with softmax output

Input = the raw feature vector in the fixed order of the handout,
e.g. x = [0, 0, 8, 9, 5, 4, 1] for a student asking for AI lab support
with severity 8, time sensitivity 9, crowd 5, distance 4, eligible.
Both networks scale x to [0, 1] internally before using it.

Both networks are written from scratch with numpy. The training data is
a synthetic campus dataset generated and stored with pandas.

IMPORTANT RULE: the ANN predicts PRIORITY ONLY. It never grants
permission and never replaces the Logic / KB validation.
-----------------------------------------------------------------------
"""

import os

import numpy as np
import pandas as pd

import config


class ModelError(Exception):
    """Raised for invalid model input or untrained models."""


# -------------------------------------------------------------------
# Synthetic dataset
# -------------------------------------------------------------------
CATEGORY_URGENCY = {
    "AI_Lab_Support": 0.5, "Viva_Scheduling": 0.3, "Access_Request": 0.2,
    "Maintenance": 0.4, "Emergency_Help": 1.0,
}
ROLE_URGENCY = {"student": 0.3, "instructor": 0.6, "staff": 0.5}
# Range of every raw feature x1..x7, used to scale x to [0, 1]
FEATURE_MIN = np.array([0, 0, config.LEVEL_MIN, config.LEVEL_MIN, config.LEVEL_MIN, 0, 0], dtype=float)
FEATURE_MAX = np.array([len(config.VALID_ROLES) - 1, len(config.CATEGORIES) - 1, config.LEVEL_MAX,
                        config.LEVEL_MAX, config.LEVEL_MAX, config.MAX_ROUTE_DISTANCE, 1], dtype=float)
PRIORITY_THRESHOLDS = (0.65, 0.50, 0.35)       # Urgent >= .65 > High >= .50 > Normal >= .35 > Low


def scale_features(raw_features):
    """
    Function: scale_features
    Purpose : Min-max scale raw feature vectors (one per row) to [0, 1] so
              that no feature dominates just because of its unit.
    """
    raw_features = np.asarray(raw_features, dtype=float)
    return np.clip((raw_features - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN), 0.0, 1.0)


def generate_dataset(number_of_samples=1500, random_seed=config.RANDOM_SEED, save_path=None):
    """
    Function: generate_dataset
    Purpose : Create a synthetic but realistic campus request dataset.
              The hidden urgency score is a weighted sum of the inputs
              (severity and time sensitivity matter most) plus Gaussian
              noise; thresholds turn it into four priority classes.
    Returns : pandas DataFrame with raw columns, raw feature vector x1..x7,
              'priority' (class name), 'priority_index' and 'is_urgent'
    """
    if number_of_samples < 50:
        raise ModelError("Use at least 50 samples to train the ANN.")
    generator = np.random.default_rng(random_seed)
    roles = generator.choice(config.VALID_ROLES, size=number_of_samples, p=[0.6, 0.25, 0.15])
    categories = generator.choice(config.CATEGORIES, size=number_of_samples)
    severity = generator.integers(config.LEVEL_MIN, config.LEVEL_MAX + 1, size=number_of_samples)
    time_sensitivity = generator.integers(config.LEVEL_MIN, config.LEVEL_MAX + 1, size=number_of_samples)
    crowd_level = generator.integers(config.LEVEL_MIN, config.LEVEL_MAX + 1, size=number_of_samples)
    distance = np.round(generator.uniform(0, config.MAX_ROUTE_DISTANCE, size=number_of_samples), 1)
    eligibility = generator.integers(0, 2, size=number_of_samples)

    data_frame = pd.DataFrame({
        "role": roles, "category": categories, "severity": severity,
        "time_sensitivity": time_sensitivity, "crowd_level": crowd_level,
        "distance": distance, "eligibility": eligibility,
    })
    raw_features = encode_dataframe(data_frame)
    features = scale_features(raw_features)
    urgency_score = (0.35 * features[:, 2] + 0.25 * features[:, 3] + 0.10 * features[:, 4]
                     + 0.10 * features[:, 5]
                     + 0.08 * data_frame["category"].map(CATEGORY_URGENCY).to_numpy()
                     + 0.07 * data_frame["role"].map(ROLE_URGENCY).to_numpy()
                     + 0.05 * features[:, 6]
                     + generator.normal(0, 0.03, size=number_of_samples))
    priority_index = np.select(
        [urgency_score >= PRIORITY_THRESHOLDS[0], urgency_score >= PRIORITY_THRESHOLDS[1],
         urgency_score >= PRIORITY_THRESHOLDS[2]], [0, 1, 2], default=3)

    for column_index, feature_name in enumerate(config.ANN_FEATURE_NAMES):
        data_frame[f"x{column_index + 1}_{feature_name}"] = np.round(raw_features[:, column_index], 2)
    data_frame["urgency_score"] = np.round(urgency_score, 4)
    data_frame["priority_index"] = priority_index
    data_frame["priority"] = [config.PRIORITY_CLASSES[i] for i in priority_index]
    data_frame["is_urgent"] = (priority_index <= 1).astype(int)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data_frame.to_csv(save_path, index=False)
    return data_frame


def encode_dataframe(data_frame):
    """
    Function: encode_dataframe
    Purpose : Convert request columns into the raw numeric feature matrix
              in the fixed order x1 Role ... x7 Eligibility, using the
              handout encodings (student = 0, AI_Lab_Support = 0, ...).
    """
    role_codes = data_frame["role"].map(config.ROLE_ENCODING)
    category_codes = data_frame["category"].map(config.REQUEST_TYPE_ENCODING)
    if role_codes.isna().any() or category_codes.isna().any():
        raise ModelError("Dataset contains an unknown role or category.")
    return np.column_stack([
        role_codes.to_numpy(dtype=float),
        category_codes.to_numpy(dtype=float),
        data_frame["severity"].to_numpy(dtype=float),
        data_frame["time_sensitivity"].to_numpy(dtype=float),
        data_frame["crowd_level"].to_numpy(dtype=float),
        data_frame["distance"].to_numpy(dtype=float),
        data_frame["eligibility"].to_numpy(dtype=float),
    ])


def train_test_split(features, labels, test_ratio=0.2, random_seed=config.RANDOM_SEED):
    """
    Function: train_test_split
    Purpose : Shuffle the data with numpy and split it into train/test parts.
    """
    generator = np.random.default_rng(random_seed)
    order = generator.permutation(len(features))
    test_count = int(len(features) * test_ratio)
    test_indices, train_indices = order[:test_count], order[test_count:]
    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]


def confusion_matrix(true_labels, predicted_labels, number_of_classes):
    """
    Function: confusion_matrix
    Purpose : Count (true, predicted) pairs into a square numpy matrix.
    """
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


# -------------------------------------------------------------------
# Perceptron (binary baseline)
# -------------------------------------------------------------------
class Perceptron:
    """
    Class: Perceptron
    Purpose: Single neuron with step activation trained with the classic
             perceptron learning rule: w <- w + lr * (y - y_hat) * x
    """

    def __init__(self, number_of_inputs=7, learning_rate=0.05, epochs=60,
                 random_seed=config.RANDOM_SEED):
        """
        Function: __init__
        Purpose : Initialise small random weights and a zero bias.
        """
        generator = np.random.default_rng(random_seed)
        self.weights = generator.normal(0, 0.1, size=number_of_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_per_epoch = []
        self.trained = False

    @staticmethod
    def step(net_input):
        """
        Function: step
        Purpose : Step activation - output 1 if net input >= 0 else 0.
        """
        return (net_input >= 0).astype(int)

    def fit(self, features, labels):
        """
        Function: fit
        Purpose : Train with the perceptron rule for a fixed number of
                  epochs, recording the misclassifications per epoch.
        """
        generator = np.random.default_rng(config.RANDOM_SEED)
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            mistakes = 0
            for index in generator.permutation(len(features)):
                prediction = self.step(np.dot(features[index], self.weights) + self.bias)
                update = self.learning_rate * (labels[index] - prediction)
                if update != 0:
                    self.weights += update * features[index]
                    self.bias += update
                    mistakes += 1
            self.errors_per_epoch.append(mistakes)
        self.trained = True
        return self

    def predict(self, features):
        """
        Function: predict
        Purpose : Predict 0/1 for one sample or a batch of samples.
        """
        features = np.atleast_2d(features)
        return self.step(features @ self.weights + self.bias)

    def save(self, path):
        """
        Function: save
        Purpose : Store weights, bias and training history in a .npz file.
        """
        np.savez(path, weights=self.weights, bias=self.bias,
                 errors=np.array(self.errors_per_epoch))

    def load(self, path):
        """
        Function: load
        Purpose : Restore a previously saved perceptron.
        """
        stored = np.load(path)
        self.weights, self.bias = stored["weights"], float(stored["bias"])
        self.errors_per_epoch = stored["errors"].tolist()
        self.trained = True
        return self


# -------------------------------------------------------------------
# Multi-Layer Perceptron (final multiclass model)
# -------------------------------------------------------------------
class MultiLayerPerceptron:
    """
    Class: MultiLayerPerceptron
    Purpose: Fully connected network 7 -> 4 -> 3 -> 4 with tanh hidden
             layers, softmax output, cross-entropy loss and mini-batch
             gradient descent with momentum (backpropagation by hand).
    """

    def __init__(self, layer_sizes=(7,) + config.MLP_HIDDEN_LAYERS + (4,), learning_rate=0.01,
                 epochs=500, batch_size=32, momentum=0.9, random_seed=config.RANDOM_SEED):
        """
        Function: __init__
        Purpose : Xavier-initialise weights and zero biases for every layer.
        """
        generator = np.random.default_rng(random_seed)
        self.layer_sizes = tuple(layer_sizes)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(generator.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.history = {"loss": [], "train_accuracy": [], "test_accuracy": []}
        self.trained = False

    @staticmethod
    def softmax(logits):
        """
        Function: softmax
        Purpose : Numerically stable softmax over the last axis.
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def forward(self, features):
        """
        Function: forward
        Purpose : Forward pass. Returns the list of layer activations
                  (input first, softmax probabilities last).
        """
        activations = [features]
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            net_input = activations[-1] @ weight + bias
            if layer_index == len(self.weights) - 1:
                activations.append(self.softmax(net_input))
            else:
                activations.append(np.tanh(net_input))
        return activations

    def _backward(self, activations, one_hot_labels):
        """
        Function: _backward
        Purpose : Backpropagation. For softmax + cross-entropy the output
                  error is (probabilities - targets); tanh derivative is
                  1 - a^2. Returns gradients for weights and biases.
        """
        batch_count = one_hot_labels.shape[0]
        delta = (activations[-1] - one_hot_labels) / batch_count
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for layer_index in reversed(range(len(self.weights))):
            weight_gradients[layer_index] = activations[layer_index].T @ delta
            bias_gradients[layer_index] = delta.sum(axis=0)
            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * (1 - activations[layer_index] ** 2)
        return weight_gradients, bias_gradients

    def fit(self, features, labels, test_features=None, test_labels=None):
        """
        Function: fit
        Purpose : Train with mini-batch gradient descent + momentum and keep
                  loss / accuracy history for plotting.
        """
        number_of_classes = self.layer_sizes[-1]
        one_hot = np.eye(number_of_classes)[labels]
        generator = np.random.default_rng(config.RANDOM_SEED)
        weight_velocity = [np.zeros_like(w) for w in self.weights]
        bias_velocity = [np.zeros_like(b) for b in self.biases]
        self.history = {"loss": [], "train_accuracy": [], "test_accuracy": []}
        for _ in range(self.epochs):
            order = generator.permutation(len(features))
            for start in range(0, len(features), self.batch_size):
                batch = order[start:start + self.batch_size]
                activations = self.forward(features[batch])
                weight_gradients, bias_gradients = self._backward(activations, one_hot[batch])
                for i in range(len(self.weights)):
                    weight_velocity[i] = self.momentum * weight_velocity[i] - self.learning_rate * weight_gradients[i]
                    bias_velocity[i] = self.momentum * bias_velocity[i] - self.learning_rate * bias_gradients[i]
                    self.weights[i] += weight_velocity[i]
                    self.biases[i] += bias_velocity[i]
            probabilities = self.forward(features)[-1]
            loss = -np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-12))
            self.history["loss"].append(float(loss))
            self.history["train_accuracy"].append(float(np.mean(probabilities.argmax(axis=1) == labels)))
            if test_features is not None:
                self.history["test_accuracy"].append(float(np.mean(self.predict(test_features) == test_labels)))
        self.trained = True
        return self

    def predict_proba(self, features):
        """
        Function: predict_proba
        Purpose : Class probabilities for one sample or a batch.
        """
        return self.forward(np.atleast_2d(np.asarray(features, dtype=float)))[-1]

    def predict(self, features):
        """
        Function: predict
        Purpose : Index of the most probable class for each sample.
        """
        return self.predict_proba(features).argmax(axis=1)

    def save(self, path):
        """
        Function: save
        Purpose : Store all layer weights, biases and history in a .npz file.
        """
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        arrays["layer_sizes"] = np.array(self.layer_sizes)
        arrays["loss"] = np.array(self.history["loss"])
        arrays["train_accuracy"] = np.array(self.history["train_accuracy"])
        arrays["test_accuracy"] = np.array(self.history["test_accuracy"])
        np.savez(path, **arrays)

    def load(self, path):
        """
        Function: load
        Purpose : Restore a previously saved MLP.
        """
        stored = np.load(path)
        self.layer_sizes = tuple(int(size) for size in stored["layer_sizes"])
        layer_count = len(self.layer_sizes) - 1
        self.weights = [stored[f"W{i}"] for i in range(layer_count)]
        self.biases = [stored[f"b{i}"] for i in range(layer_count)]
        self.history = {key: stored[key].tolist() for key in ("loss", "train_accuracy", "test_accuracy")}
        self.trained = True
        return self


# -------------------------------------------------------------------
# Priority predictor that wraps both networks
# -------------------------------------------------------------------
class PriorityPredictor:
    """
    Class: PriorityPredictor
    Purpose: Train / load the Perceptron and the MLP and predict the
             priority of a single preprocessed request.
    """

    def __init__(self):
        """
        Function: __init__
        Purpose : Create untrained models and an empty metrics dict.
        """
        self.perceptron = Perceptron()
        self.mlp = MultiLayerPerceptron()
        self.metrics = {}

    def train(self, number_of_samples=1500, verbose=True):
        """
        Function: train
        Purpose : Generate the dataset, train both networks, evaluate them
                  on a held-out test set and save weights to disk.
        Returns : metrics dict (accuracies, confusion matrices, sizes)
        """
        config.ensure_directories()
        data_frame = generate_dataset(number_of_samples, save_path=config.ANN_DATASET_PATH)
        features = scale_features(data_frame[[f"x{i + 1}_{n}" for i, n in enumerate(config.ANN_FEATURE_NAMES)]]
                                  .to_numpy())
        multiclass_labels = data_frame["priority_index"].to_numpy()
        binary_labels = data_frame["is_urgent"].to_numpy()

        train_x, test_x, train_y, test_y = train_test_split(features, multiclass_labels)
        _, _, train_b, test_b = train_test_split(features, binary_labels)

        if verbose:
            print(f"  Dataset: {len(data_frame)} samples saved to {config.ANN_DATASET_PATH}")
            print("  Training Perceptron (binary baseline)...")
        self.perceptron = Perceptron().fit(train_x, train_b)
        perceptron_predictions = self.perceptron.predict(test_x)
        if verbose:
            print("  Training MLP 7-4-3-4 (multiclass)...")
        self.mlp = MultiLayerPerceptron().fit(train_x, train_y, test_x, test_y)
        mlp_predictions = self.mlp.predict(test_x)

        self.metrics = {
            "samples": len(data_frame),
            "train_size": len(train_x),
            "test_size": len(test_x),
            "perceptron_accuracy": float(np.mean(perceptron_predictions == test_b)),
            "perceptron_confusion": confusion_matrix(test_b, perceptron_predictions, 2),
            "mlp_accuracy": float(np.mean(mlp_predictions == test_y)),
            "mlp_confusion": confusion_matrix(test_y, mlp_predictions, len(config.PRIORITY_CLASSES)),
            "class_distribution": data_frame["priority"].value_counts().reindex(config.PRIORITY_CLASSES, fill_value=0),
        }
        self.perceptron.save(config.PERCEPTRON_WEIGHTS_PATH)
        self.mlp.save(config.MLP_WEIGHTS_PATH)
        return self.metrics

    def load_or_train(self, verbose=False):
        """
        Function: load_or_train
        Purpose : Load saved weights if both files exist and are readable;
                  otherwise train new models.
        """
        try:
            if os.path.exists(config.PERCEPTRON_WEIGHTS_PATH) and os.path.exists(config.MLP_WEIGHTS_PATH):
                self.perceptron.load(config.PERCEPTRON_WEIGHTS_PATH)
                self.mlp.load(config.MLP_WEIGHTS_PATH)
                return self
        except (OSError, KeyError, ValueError) as error:
            if verbose:
                print(f"  Saved models unreadable ({error}); retraining.")
        self.train(verbose=verbose)
        return self

    def predict(self, feature_vector):
        """
        Function: predict
        Purpose : Predict the priority of one raw request feature vector
                  x = [Role, RequestType, Severity, TimeSensitivity,
                       CrowdLevel, Distance, Eligibility].
        Returns : standard priority output {binary_priority (Perceptron),
                  final_priority (MLP), confidence} plus the class
                  probabilities and the feature vector for explanations
        """
        if not (self.mlp.trained and self.perceptron.trained):
            raise ModelError("ANN models are not trained. Train them first (menu option 4).")
        raw_vector = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        if raw_vector.shape[1] != len(config.ANN_FEATURE_NAMES):
            raise ModelError(f"Expected {len(config.ANN_FEATURE_NAMES)} features, "
                             f"got {raw_vector.shape[1]}.")
        scaled_vector = scale_features(raw_vector)
        probabilities = self.mlp.predict_proba(scaled_vector)[0]
        predicted_index = int(np.argmax(probabilities))
        perceptron_urgent = bool(self.perceptron.predict(scaled_vector)[0])
        return {
            **config.PRIORITY_OUTPUT_TEMPLATE,
            "binary_priority": config.BINARY_CLASSES[int(perceptron_urgent)],
            "final_priority": config.PRIORITY_CLASSES[predicted_index],
            "confidence": round(float(probabilities[predicted_index]), 2),
            "probabilities": {name: round(float(p), 3)
                              for name, p in zip(config.PRIORITY_CLASSES, probabilities)},
            "features": dict(zip(config.ANN_FEATURE_NAMES, [float(value) for value in raw_vector[0]])),
            "scaled_features": dict(zip(config.ANN_FEATURE_NAMES, np.round(scaled_vector[0], 3).tolist())),
        }
