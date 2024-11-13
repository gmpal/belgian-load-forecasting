import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, embedded_vector_size, non_linear_vector_size, device=None):
        super(SimpleMLP, self).__init__()
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.l_nl = nn.Linear(input_size, non_linear_vector_size).to(self.device)
        self.l_l = nn.Linear(input_size, embedded_vector_size).to(self.device)
        self.l = nn.ModuleList()
        self.l.append(nn.Linear(non_linear_vector_size + input_size + embedded_vector_size, embedded_vector_size + 1).to(self.device))
        for _ in range(1, output_size + 1):
            self.l.append(nn.Linear(non_linear_vector_size + input_size + 2 * embedded_vector_size + 1, embedded_vector_size + 1).to(self.device))
        self.tanh = nn.Tanh()

    def forward(self, x):
        outputs = []
        x_linear = self.l_l(x)
        x_non_linear = self.tanh(self.l_nl(x))

        xs = [self.l[0](torch.cat([x, x_linear, x_non_linear], dim=-1))]
        for i in range(1, len(self.l)):
            xs.append(self.l[i](torch.cat([x, x_linear, x_non_linear, xs[-1]], dim=-1)))
            outputs.append(xs[-1][:, -1:])
        out = torch.cat(outputs, dim=-1)
        return out

class SimpleMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=None, output_size=None, embedded_vector_size=None, non_linear_vector_size=None, batch_size=64, epochs=100, lr=0.001, device=None, verbose=False):
        self.input_size = input_size
        self.output_size = output_size
        self.embedded_vector_size = embedded_vector_size
        self.non_linear_vector_size = non_linear_vector_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

    def fit(self, X, y):
        # Convert X and y to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Determine input_size and output_size if not set
        if self.input_size is None:
            self.input_size = X.shape[1]
        if self.output_size is None:
            self.output_size = y.shape[1] if len(y.shape) > 1 else 1

        # Set default values for embedded_vector_size and non_linear_vector_size if not set
        if self.embedded_vector_size is None:
            self.embedded_vector_size = self.input_size // 4
        if self.non_linear_vector_size is None:
            self.non_linear_vector_size = self.input_size

        # Create the dataset and dataloader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the model
        self.model = SimpleMLP(self.input_size, self.output_size, self.embedded_vector_size, self.non_linear_vector_size, device=self.device).to(self.device)

        # Define the criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training epochs", disable=not self.verbose):
            epoch_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if self.verbose:
                avg_loss = epoch_loss / num_batches
                # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        return self

    def predict(self, X):
        # Convert X to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Create DataLoader
        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Evaluate the model
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).cpu()
                predictions.append(outputs)

        predictions = torch.cat(predictions, dim=0).numpy()
        return predictions

