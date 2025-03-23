"""Model training and optimization module for HelixZone."""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Protocol, cast, TYPE_CHECKING, Union, Type
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy.typing as npt
from .gpu_manager import GPUResourceMonitor
from .logging_manager import LoggingManager

if TYPE_CHECKING:
    from torch.cuda import Stream
    from torch.cuda.amp import GradScaler

# Type aliases
Array = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float32]
CudaEvent = torch.cuda.Event
ModelClass = Type[nn.Module]

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    weight_decay: float = 0.0001
    gradient_clip_val: float = 1.0

@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    best_epoch: int
    training_time: float

class ModelTrainer:
    """Handles model training with optimization and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        logger: Optional[LoggingManager] = None
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.logger = logger or LoggingManager()
        self.gpu_monitor = GPUResourceMonitor()
        self.scaler: GradScaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(
        self,
        X: Array,
        y: Array,
        validation_data: Optional[Tuple[Array, Array]] = None
    ) -> TrainingMetrics:
        """Train the model with validation and early stopping."""
        with self.logger.track_operation("model_training"):
            # Convert data to tensors
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_tensor = torch.from_numpy(y).float().to(self.device)
            
            # Split data if validation_data not provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_tensor,
                    y_tensor,
                    test_size=self.config.validation_split,
                    random_state=42
                )
            else:
                X_train, y_train = X_tensor, y_tensor
                X_val = torch.from_numpy(validation_data[0]).float().to(self.device)
                y_val = torch.from_numpy(validation_data[1]).float().to(self.device)
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Initialize metrics tracking
            metrics = TrainingMetrics(
                train_loss=[],
                val_loss=[],
                train_accuracy=[],
                val_accuracy=[],
                best_epoch=0,
                training_time=0.0
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Create CUDA events for timing
            cuda_start_time: Optional[CudaEvent] = None
            cuda_end_time: Optional[CudaEvent] = None
            
            if torch.cuda.is_available():
                current_stream: Stream = torch.cuda.current_stream()
                cuda_start_time = torch.cuda.Event(enable_timing=True)
                cuda_end_time = torch.cuda.Event(enable_timing=True)
                cuda_start_time.record(current_stream)
            
            try:
                for epoch in range(self.config.num_epochs):
                    # Training phase
                    self.model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for i in range(0, len(X_train), self.config.batch_size):
                        batch_X = X_train[i:i + self.config.batch_size]
                        batch_y = y_train[i:i + self.config.batch_size]
                        
                        optimizer.zero_grad()
                        
                        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                            outputs = self.model(batch_X)
                            loss = F.mse_loss(outputs, batch_y)
                            
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Clip gradients
                        if self.config.gradient_clip_val > 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.gradient_clip_val
                            )
                            
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        
                        train_loss += loss.item()
                        
                        # Calculate accuracy
                        predicted = outputs.round()
                        train_total += batch_y.size(0)
                        train_correct += (predicted == batch_y).sum().item()
                        
                    # Validation phase
                    self.model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for i in range(0, len(X_val), self.config.batch_size):
                            batch_X = X_val[i:i + self.config.batch_size]
                            batch_y = y_val[i:i + self.config.batch_size]
                            
                            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                                outputs = self.model(batch_X)
                                loss = F.mse_loss(outputs, batch_y)
                                
                            val_loss += loss.item()
                            
                            # Calculate accuracy
                            predicted = outputs.round()
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()
                            
                    # Update metrics
                    avg_train_loss = train_loss / (len(X_train) / self.config.batch_size)
                    avg_val_loss = val_loss / (len(X_val) / self.config.batch_size)
                    train_accuracy = train_correct / train_total
                    val_accuracy = val_correct / val_total
                    
                    metrics.train_loss.append(avg_train_loss)
                    metrics.val_loss.append(avg_val_loss)
                    metrics.train_accuracy.append(train_accuracy)
                    metrics.val_accuracy.append(val_accuracy)
                    
                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        metrics.best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                        
            except Exception as e:
                self.logger.error_logger.log_error(e, {"phase": "training"})
                raise
                
            finally:
                if torch.cuda.is_available() and cuda_start_time is not None and cuda_end_time is not None:
                    cuda_end_time.record(torch.cuda.current_stream())
                    torch.cuda.synchronize()
                    metrics.training_time = cuda_start_time.elapsed_time(cuda_end_time) / 1000  # Convert to seconds
                else:
                    metrics.training_time = 0.0  # Fallback when CUDA is not available
                
            return metrics
            
    def predict(self, X: Array) -> Array:
        """Make predictions using the trained model."""
        with self.logger.track_operation("model_prediction"):
            self.model.eval()
            X_tensor = torch.from_numpy(X).float().to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    predictions = self.model(X_tensor)
                    
            return predictions.cpu().numpy()
            
    def evaluate(self, X: Array, y: Array) -> Dict[str, float]:
        """Evaluate model performance."""
        with self.logger.track_operation("model_evaluation"):
            self.model.eval()
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_tensor = torch.from_numpy(y).float().to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(X_tensor)
                    loss = F.mse_loss(outputs, y_tensor)
                    
                    # Calculate accuracy
                    predicted = outputs.round()
                    correct = (predicted == y_tensor).sum().item()
                    total = y_tensor.size(0)
                    
            return {
                'loss': loss.item(),
                'accuracy': correct / total
            }
            
    def save_model(self, filepath: str) -> None:
        """Save model state."""
        with self.logger.track_operation("model_saving"):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, filepath)
            
    def load_model(self, filepath: str) -> None:
        """Load model state."""
        with self.logger.track_operation("model_loading"):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.config = checkpoint['config']
            self.model.to(self.device)

class ModelOptimizer:
    """Optimizes model hyperparameters."""
    
    def __init__(
        self,
        model_class: type,
        logger: Optional[LoggingManager] = None
    ) -> None:
        self.model_class = model_class
        self.logger = logger or LoggingManager()
        self.gpu_monitor = GPUResourceMonitor()
        
    def optimize(
        self,
        X: Array,
        y: Array,
        param_grid: Dict[str, List[Any]],
        n_trials: int = 10
    ) -> Tuple[Dict[str, Any], TrainingMetrics]:
        """Optimize hyperparameters using random search."""
        with self.logger.track_operation("hyperparameter_optimization"):
            best_params = {}
            best_metrics = None
            best_val_loss = float('inf')
            
            for trial in range(n_trials):
                # Sample parameters
                params = {
                    key: np.random.choice(values)
                    for key, values in param_grid.items()
                }
                
                # Create model and trainer
                model = self.model_class(**params)
                config = TrainingConfig(
                    learning_rate=params.get('learning_rate', 0.001),
                    batch_size=params.get('batch_size', 32),
                    num_epochs=params.get('num_epochs', 100)
                )
                trainer = ModelTrainer(model, config, self.logger)
                
                # Train model
                metrics = trainer.train(X, y)
                
                # Update best parameters if better
                if min(metrics.val_loss) < best_val_loss:
                    best_val_loss = min(metrics.val_loss)
                    best_params = params
                    best_metrics = metrics
                    
            return best_params, cast(TrainingMetrics, best_metrics) 