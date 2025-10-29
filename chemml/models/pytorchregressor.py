import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class PyTorchRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_functions, dropout_rate=0.3):
        super(PyTorchRegressor, self).__init__()
        self.layers = nn.ModuleList()
        
        activation_map = {
            'leakyrelu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(activation_map[activation_functions[i]])
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            
        # Output layer
        self.out = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

class PyTorchRegressionModel:
    def __init__(self, model, optimizer_choice='AdamW', lr=0.001, alpha=0.0001, epochs=200, batch_size=64, patience=20, use_cosine_lr=True):
        self.model = model
        self.optimizer_choice = optimizer_choice
        self.lr = lr
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_loss = float('inf')
        self.no_improve = 0
        self.optimizer = self.set_optimizer()
        self.use_cosine_lr = use_cosine_lr
        if use_cosine_lr:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.loss_history = []
        self.val_loss_history = []

    def set_optimizer(self):
        optimizers = {
            'AdamW': optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.alpha),
            'Adam': optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.alpha),
            'SGD': optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.alpha)
        }
        return optimizers[self.optimizer_choice]

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32), 
            torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32), 
            torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = nn.MSELoss()(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            self.loss_history.append(avg_epoch_loss)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                    outputs_val = self.model(batch_X_val)
                    loss = nn.MSELoss()(outputs_val, batch_y_val)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / max(1, len(val_loader))
            self.val_loss_history.append(avg_val_loss)

            if self.use_cosine_lr:
                self.scheduler.step()
            else:
                self.scheduler.step(avg_val_loss)

            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.no_improve = 0
            else:
                self.no_improve += 1

            if self.no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}] Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler_X.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor).cpu().numpy()
        return self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self, deep=True):
        return self.model.state_dict() if deep else {}


class PyTorchRegressorWrapper:
    def __init__(self, input_dim, hidden_layers=None, n_layers=None, base_neurons=None, 
                 output_dim=1, activation_functions=['Swish'], optimizer_choice='AdamW', 
                 lr=0.001, alpha=0.001, epochs=200, batch_size=64, 
                 dropout_rate=0.2, patience=20, use_cosine_lr=True):

        if hidden_layers is None:
            if n_layers is not None and base_neurons is not None:
                hidden_layers = [base_neurons] * n_layers
            else:
                raise ValueError("You must provide either 'hidden_layers' or both 'n_layers' and 'base_neurons'.")

        if not hidden_layers:
            raise ValueError("hidden_layers list cannot be empty.")

        activation_functions = [act.strip().lower() for act in activation_functions]
        
        required_activations = len(hidden_layers) - 1
        if len(activation_functions) < required_activations:
            activation_functions += [activation_functions[-1]] * (required_activations - len(activation_functions))
        elif len(activation_functions) > required_activations:
            activation_functions = activation_functions[:required_activations]
        
        # Check for valid activation function names after finalizing the list
        valid_activations = {'leakyrelu', 'swish', 'gelu'}
        for act in activation_functions:            
            if act not in valid_activations:                
                raise ValueError(f"Invalid activation: {act}. Valid options are 'leakyrelu', 'swish', 'gelu'.")

        self.model = PyTorchRegressor(input_dim, hidden_layers, output_dim, activation_functions, dropout_rate)
        self.regression_model = PyTorchRegressionModel(
            self.model, optimizer_choice, lr, alpha, epochs, batch_size, patience, use_cosine_lr
        )

    def fit(self, X, y):
        self.regression_model.fit(X, y)

    def predict(self, X):
        return self.regression_model.predict(X)

    def score(self, X, y):
        return self.regression_model.score(X, y)

    def get_params(self, deep=True):
        return self.regression_model.get_params(deep)
