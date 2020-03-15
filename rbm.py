import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# noinspection PyPep8Naming
class RBM(object):
    def __init__(self, visible_units: int, hidden_units: int, alpha: float, iterations: int = 1):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.iterations = iterations

        self.weights: np.ndarray = np.random.normal(loc=0, scale=0.25, size=(visible_units, hidden_units))
        self.biases_visible: np.ndarray = np.zeros(visible_units, dtype=np.float32)
        self.biases_hidden: np.ndarray = np.zeros(hidden_units, dtype=np.float32)

    def fit(self, X: np.ndarray):
        """
        Implements Contrastive Divergence Algorithm, based on book
        "Neural Networks and Deep Learning" by Charu C. Aggarwal.
        

        Parameters
        ----------
        X : array with shape [number of examples, number of features]

        """

        self._fit(X)

    def _fit(self, X):
        assert X.shape[1] == self.visible_units

        number_of_examples = X.shape[0]

        positive_hidden_units_samples = self._sample_hidden_from_visible(X)
        assert positive_hidden_units_samples.shape == (number_of_examples, self.hidden_units)

        for i in range(self.iterations):
            negative_visible_units_samples = self._sample_visible_from_hidden(positive_hidden_units_samples)
            assert negative_visible_units_samples.shape == (number_of_examples, self.visible_units)

            negative_hidden_units_samples = self._sample_hidden_from_visible(negative_visible_units_samples)
            assert negative_hidden_units_samples.shape == (number_of_examples, self.hidden_units)

        positive_correlation = (X.transpose()[:, :, np.newaxis] * positive_hidden_units_samples).mean(axis=1)
        negative_correlation = (X.transpose()[:, :, np.newaxis] * negative_hidden_units_samples).mean(axis=1)
        assert positive_correlation.shape == negative_correlation.shape == self.weights.shape

        self.weights += self.alpha * (positive_correlation - negative_correlation)
        self.biases_visible += self.alpha * (X.mean(axis=0) - negative_visible_units_samples.mean(axis=0))
        self.biases_hidden += self.alpha * (
                positive_hidden_units_samples.mean(axis=0) - negative_hidden_units_samples.mean(axis=0)
        )

    def _sample_hidden_from_visible(self, X):
        number_of_examples = X.shape[0]

        R = X @ self.weights
        assert R.shape == (number_of_examples, self.hidden_units)

        hidden_units_probability = sigmoid(self.biases_hidden + R)
        assert hidden_units_probability.shape == (number_of_examples, self.hidden_units)

        return np.random.binomial(n=1, p=hidden_units_probability)

    def _sample_visible_from_hidden(self, X):
        number_of_examples = X.shape[0]

        R = X @ self.weights.transpose()
        assert R.shape == (number_of_examples, self.visible_units)

        visible_units_probability = sigmoid(self.biases_visible + R)
        assert visible_units_probability.shape == (number_of_examples, self.visible_units)

        return np.random.binomial(n=1, p=visible_units_probability)


RBM(5, 3, alpha=0.1).fit(np.array([[0, 0, 1, 1, 0], [1, 1, 1, 0, 0]]))
