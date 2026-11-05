"""M2 neural models as replicated by Walsh et al. 2024: the ANN of Anderson et al. 2020 and the 1D-CNN of Cui & Fearn
used by Mishra & Passos 2021. Architecture and training settings follow Walsh's code
(github.com/jemjemwalsh/research-deep-learning-nirs, publications/Eval-1D-CNN-Mango-DM/utils/models.py).
Needs TensorFlow, so it runs on Colab; TensorFlow is imported lazily to keep the rest of src/ usable without it.
"""
import time

import numpy as np

BATCH = 128
LEARNING_RATE = 0.01 * BATCH / 256  # 0.005
L2_BETA = 0.011 / 2
MAX_EPOCHS = {'ann': 10000, 'cnn': 750}


def _keras():
    import tensorflow as tf
    return tf.keras


def build_ann(n_features, seed):
    # one hidden layer of 5 sigmoid neurons, linear output; default initialisers seeded through set_random_seed
    keras = _keras()
    return keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(5, activation='sigmoid'),
        keras.layers.Dense(1, activation='linear'),
    ])


def build_cnn(n_features, seed):
    # Conv1D (1 filter, width 21, stride 1) -> Dense 36, 18, 12 (ELU) -> linear output; HeNormal init and L2 on all layers
    keras = _keras()

    def kw():
        return {'kernel_initializer': keras.initializers.HeNormal(seed=seed),
                'kernel_regularizer': keras.regularizers.l2(L2_BETA)}

    return keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Reshape((n_features, 1)),
        keras.layers.Conv1D(1, 21, strides=1, padding='same', activation='elu', **kw()),
        keras.layers.Flatten(),
        keras.layers.Dense(36, activation='elu', **kw()),
        keras.layers.Dense(18, activation='elu', **kw()),
        keras.layers.Dense(12, activation='elu', **kw()),
        keras.layers.Dense(1, activation='linear', **kw()),
    ])


def make_callbacks(monitor_val):
    """Without a tuning set the training loss is monitored, with the looser early-stopping delta Walsh used."""
    keras = _keras()
    monitor = 'val_loss' if monitor_val else 'loss'
    return [
        # min_delta 1e-4 is the Keras default, as in Walsh's code (the paper text says 1e-6)
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=25, factor=0.5, min_lr=1e-6, min_delta=1e-4),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=50, min_delta=1e-5 if monitor_val else 10 ** -2.5,
                                      restore_best_weights=True),
    ]


def train(kind, X_train, y_train, X_val=None, y_val=None, seed=0, max_epochs=None, verbose=0):
    keras = _keras()
    keras.utils.set_random_seed(seed)
    model = {'ann': build_ann, 'cnn': build_cnn}[kind](X_train.shape[1], seed)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    validation = (X_val, y_val) if X_val is not None else None
    t0 = time.time()
    history = model.fit(X_train, y_train, batch_size=BATCH, epochs=max_epochs or MAX_EPOCHS[kind],
                        validation_data=validation, callbacks=make_callbacks(validation is not None),
                        shuffle=True, verbose=verbose)
    h = history.history
    monitored = h['val_loss'] if validation is not None else h['loss']
    return model, {'epochs': len(h['loss']), 'best_epoch': int(np.argmin(monitored)) + 1,
                   'train_seconds': time.time() - t0, 'best_monitored_loss': float(np.min(monitored))}


def predict(model, X):
    return model.predict(X, batch_size=4096, verbose=0).ravel()
