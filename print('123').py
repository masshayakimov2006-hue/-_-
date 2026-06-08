import numpy as np
from typing import Optional, List, Tuple

class MAPE:
    """Mean Absolute Percentage Error"""
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
        y_true = np.where(np.abs(y_true) < eps, eps, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true))
    
    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
        return -np.sign(y_true - y_pred) / (len(y_true) * y_true_safe)


class LinearRegressionGD:
    """Линейная регрессия с градиентным спуском и MAPE"""
    
    def __init__(self, lr: float = 0.01, epochs: int = 1000, tol: Optional[float] = 1e-6,
                 reg: Optional[str] = None, reg_lambda: float = 0.01, verbose: bool = False):
        self.lr, self.epochs, self.tol = lr, epochs, tol
        self.reg, self.reg_lambda = reg, reg_lambda
        self.verbose = verbose
        self.weights, self.bias, self.loss_history = None, None, []
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
    
    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        loss = MAPE.compute(y_true, y_pred)
        if self.reg == 'l1':
            loss += self.reg_lambda * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            loss += 0.5 * self.reg_lambda * np.sum(self.weights ** 2)
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).flatten()
        n_samples, n_features = X.shape
        
        # Инициализация
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # Градиенты MAPE
            grad_loss = MAPE.gradient(y, y_pred)
            grad_w = X.T @ grad_loss
            grad_b = np.sum(grad_loss)
            
            # Градиенты регуляризации
            if self.reg == 'l1':
                grad_w += self.reg_lambda * np.sign(self.weights)
            elif self.reg == 'l2':
                grad_w += self.reg_lambda * self.weights
            
            # Обновление параметров
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
            
            # Логирование
            loss = self._loss(y, y_pred)
            self.loss_history.append(loss)
            
            if self.verbose and (epoch % 200 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d} | MAPE: {loss:.6f}")
            
            # Ранняя остановка
            if self.tol and epoch > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return MAPE.compute(y, self.predict(X))


# Пример использования
if __name__ == "__main__":
    # Генерация данных
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = np.abs(X @ np.array([2, -1, 0.5]) + 1.5 + np.random.randn(200) * 0.3) + 1
    
    # Обучение вайбкодингу
    model = LinearRegressionGD(lr=0.01, epochs=500, reg='l2', reg_lambda=0.001, verbose=True)
    model.fit(X, y)
    
    print(f"\nWeights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    print(f"Test MAPE: {model.score(X, y):.6f}")