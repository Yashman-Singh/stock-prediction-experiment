import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
from typing import Tuple, Optional

class StockPredictionModel:
    """LSTM model with attention mechanism for stock prediction experiment."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 8,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        num_heads: int = 4
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build an LSTM model with attention mechanism for experimentation."""
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Bidirectional LSTM layers
        x = tf.keras.layers.Bidirectional(LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_initializer='he_normal',
            recurrent_dropout=0.1
        ))(inputs)
        x = LayerNormalization()(x)
        
        # Multi-head self attention layer
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.lstm_units // self.num_heads
        )(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Second Bidirectional LSTM layer
        x = tf.keras.layers.Bidirectional(LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            kernel_initializer='he_normal',
            recurrent_dropout=0.1
        ))(x)
        x = LayerNormalization()(x)
        
        # Dense layers with residual connections
        dense1 = Dense(64, activation='relu')(x)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        
        # Residual connection
        concat = tf.keras.layers.Concatenate()([dense1, dense2])
        
        # Final prediction layer
        outputs = Dense(1, activation='linear')(concat)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with fixed learning rate and custom loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        def custom_loss(y_true, y_pred):
            # Huber loss for robustness
            huber = tf.keras.losses.Huber(delta=1.0)
            return huber(y_true, y_pred)
        
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for training or prediction."""
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:(i + self.sequence_length)]
            sequences.append(seq)
            
            if target is not None:
                targets.append(target[i + self.sequence_length])
        
        if target is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences), None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: list = None,
        **kwargs
    ):
        """Train the model with early stopping and learning rate reduction."""
        
        # Default callbacks for model optimization
        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Combine default callbacks with any user-provided callbacks
        if callbacks:
            default_callbacks.extend(callbacks)
        
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            **kwargs
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and return various metrics."""
        predictions = self.predict(X_test)
        
        # Create metric objects
        mse_metric = tf.keras.metrics.MeanSquaredError()
        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()
        
        # Update metrics
        mse_metric.update_state(y_test, predictions)
        mae_metric.update_state(y_test, predictions)
        mape_metric.update_state(y_test, predictions)
        
        # Calculate R² manually
        y_mean = np.mean(y_test)
        r2 = 1 - np.sum((y_test - predictions.flatten())**2) / np.sum((y_test - y_mean)**2)
        
        metrics = {
            'mse': float(mse_metric.result().numpy()),
            'mae': float(mae_metric.result().numpy()),
            'mape': float(mape_metric.result().numpy()),
            'r2': float(r2)
        }
        
        return metrics 