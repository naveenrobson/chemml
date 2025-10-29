import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import tensorflow_addons as tfa
    HAS_ADAMW = True
except ImportError:
    HAS_ADAMW = False

# Map activation names to TensorFlow functions
def get_tf_activation(name):
    name = name.lower()
    valid_activations = {
        'swish': tf.nn.swish,
        'gelu': tf.nn.gelu,
        'leakyrelu': tf.nn.leaky_relu
    }
    if name not in valid_activations:
        raise ValueError(f"Unsupported activation: {name}. Valid options: {list(valid_activations.keys())}")
    return valid_activations[name]

class TensorFlowRegressionModel:
    def __init__(self, input_dim, hidden_layers, activation_functions,
                 lr=0.001, alpha=0.0001, epochs=200, optimizer_choice='AdamW',
                 batch_size=64, dropout_rate=0.3, patience=20):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions
        self.lr = lr
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer_choice = optimizer_choice
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,  
                activation=get_tf_activation(self.activation_functions[i]),              
                kernel_regularizer=regularizers.l2(self.alpha)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
            
        # Output layer
        
        model.add(layers.Dense(1))

        # Optimizer
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr,
            decay_steps=self.epochs
        )
        if self.optimizer_choice.lower() == "adamw" and HAS_ADAMW:
            optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=self.alpha, global_clipnorm=5.0)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5.0)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.5),  # Robust loss
            metrics=["mae"]
        )
        return model

    def fit(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        # Calculate validation split size
        total_samples = X_scaled.shape[0]
        val_split = max(1, int(0.2 * total_samples))  # Ensure ≥1 validation sample
        train_size = total_samples - val_split
        # Split first, then batch
        full_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, y_scaled))
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)
            
        # Dynamic batch size adjustment
        if self.batch_size > train_size:
            self.batch_size = max(1, train_size)  # Prevent batch_size > samples
        # Batch after splitting
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(self.batch_size)
    

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        if len(list(train_dataset)) == 0:
            raise ValueError(f"Training data empty. Adjust batch_size (current: {self.batch_size})")
        self.model = self._build_model() 
        history = self.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=self.epochs,
        callbacks=[early_stop],
        verbose=0)
        self.history = history  # Save for outside access if needed
        return history


    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "activation_functions": self.activation_functions,
            "lr": self.lr,
            "alpha": self.alpha,
            "epochs": self.epochs,
            "optimizer_choice": self.optimizer_choice,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
            "patience": self.patience
        }

class TensorFlowRegressorWrapper:
    def __init__(self, input_dim, hidden_layers, output_dim=1,
                 activation_functions=['swish'], optimizer_choice='AdamW',
                 lr=0.001, alpha=0.001, epochs=200, batch_size=64,
                 dropout_rate=0.2, patience=20):
        
        # Handle genetic algorithm's character list format
        if isinstance(activation_functions, list) and all(len(a) == 1 for a in activation_functions):
            activation_functions = [''.join(activation_functions)] * len(hidden_layers)
        
        # Ensure list format and correct length
        if isinstance(activation_functions, str):
            activation_functions = [activation_functions]
            
        n_layers = len(hidden_layers)
        if len(activation_functions) < n_layers:
            activation_functions += [activation_functions[-1]] * (n_layers - len(activation_functions))
        elif len(activation_functions) > n_layers:
            activation_functions = activation_functions[:n_layers]
        self.regression_model = TensorFlowRegressionModel(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation_functions=activation_functions,
            lr=lr,
            alpha=alpha,
            epochs=epochs,
            optimizer_choice=optimizer_choice,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            patience=patience
        )

    def fit(self, X, y):                                
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Empty training data received")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different sample counts")
        self.regression_model.fit(X, y)

    def predict(self, X):
        return self.regression_model.predict(X)

    def score(self, X, y):
        return self.regression_model.score(X, y)

    def get_params(self, deep=True):
        return self.regression_model.get_params(deep=deep)
