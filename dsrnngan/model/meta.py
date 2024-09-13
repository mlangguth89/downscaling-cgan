import pickle
from tensorflow.keras import backend as K


class Nontrainable(object):

    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        self.models = models

    def __enter__(self):
        self.trainable_status = [m.trainable for m in self.models]
        for m in self.models:
            m.trainable = False
        return self.models

    def __exit__(self, type, value, traceback):
        for (m, t) in zip(self.models, self.trainable_status):
            m.trainable = t


def save_opt_weights(model, filepath):
    """
    Save state of optimizer to pickle-file for later model loading/restoration (to resume training)
    :param model: The compiled Keras model object with optimizer after training
    :param filepath: Path to pickle-file to save optimizer state
    """
    filepath = filepath if filepath.endswith(".pkl") else f"{filepath}.pkl"
    
    weight_values = [v.numpy() for v in model.optimizer.variables()]

    if weight_values:
        with open(filepath, "wb") as f:
            pickle.dump(weight_values, f)

    else:
        raise ValueError(f"Failed to deduce optimizer weight from {model.optimizer}")


def load_opt_weights(model, filepath):
    """
    Load optimizer state from pickle-file (as saved with save_opt_weights-method)
    :param model: The compile Keras model whose optimizer weights should be set
    :param filepath: Path to pickle-file from which optimizer state is read
    """
    filepath = filepath if filepath.endswith(".pkl") else f"{filepath}.pkl"

    with open(filepath, "rb") as f:
        weight_values = pickle.load(f)

    model.optimizer.set_weights(weight_values)


def ensure_list(x):
    if type(x) != list:
        x = [x]
    return x


def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in
              model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes
