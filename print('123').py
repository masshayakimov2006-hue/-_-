"""
gradient_descent_wape.py - Краткая библиотека градиентного спуска с WAPE
"""

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

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100


class LinearRegressionGD:
    """Линейная регрессия с градиентным спуском и MAPE"""
    """Линейная регрессия с градиентным спуском и оптимизацией по WAPE"""

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
    def __init__(self, lr=0.01, epochs=1000, batch_size=None, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).flatten()
        n_samples, n_features = X.shape
    def fit(self, X, y, verbose=False):
        X, y = np.array(X), np.array(y).flatten()
        n, m = X.shape

        # Инициализация
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.weights = np.zeros(m)
        self.bias = 0.0
        self.loss_history = []
        batch_size = self.batch_size or n
        
        prev_loss = np.inf

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # Градиенты MAPE
            grad_loss = MAPE.gradient(y, y_pred)
            grad_w = X.T @ grad_loss
            grad_b = np.sum(grad_loss)
            # Перемешивание и разбиение на батчи
            idx = np.random.permutation(n)
            X_shuffled, y_shuffled = X[idx], y[idx]

            # Градиенты регуляризации
            if self.reg == 'l1':
                grad_w += self.reg_lambda * np.sign(self.weights)
            elif self.reg == 'l2':
                grad_w += self.reg_lambda * self.weights
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Предсказание и градиенты
                y_pred = X_batch @ self.weights + self.bias
                denominator = np.sum(np.abs(y_batch))
                
                if denominator > 0:
                    grad_abs = np.sign(y_pred - y_batch)
                    dw = (X_batch.T @ grad_abs) / denominator
                    db = np.sum(grad_abs) / denominator
                    
                    # Обновление параметров
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db

            # Обновление параметров
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
            
            # Логирование
            loss = self._loss(y, y_pred)
            # Мониторинг
            y_pred_all = X @ self.weights + self.bias
            loss = wape(y, y_pred_all)
            self.loss_history.append(loss)

            if self.verbose and (epoch % 200 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d} | MAPE: {loss:.6f}")
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, WAPE: {loss:.4f}%")

            # Ранняя остановка
            if self.tol and epoch > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"Сошлось на эпохе {epoch+1}")
                break
            prev_loss = loss

        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return MAPE.compute(y, self.predict(X))
    def predict(self, X):
        return np.array(X) @ self.weights + self.bias
    
    def score(self, X, y):
        return wape(y, self.predict(X))


# Пример использования
if __name__ == "__main__":
    # Генерация данных
    # Данные
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = np.abs(X @ np.array([2, -1, 0.5]) + 1.5 + np.random.randn(200) * 0.3) + 1
    X = np.random.randn(100, 2)
    y = 3*X[:, 0] + 2*X[:, 1] + 1 + np.random.randn(100) * 0.1

    # Обучение вайбкодингу
    model = LinearRegressionGD(lr=0.01, epochs=500, reg='l2', reg_lambda=0.001, verbose=True)
    model.fit(X, y)
    # Обучение
    model = LinearRegressionGD(lr=0.05, epochs=500, batch_size=20)
    model.fit(X, y, verbose=True)

    print(f"\nWeights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    print(f"Test MAPE: {model.score(X, y):.6f}")
    # Результаты
    print(f"\nВеса: {model.weights}")
    print(f"Смещение: {model.bias:.4f}")
    print(f"WAPE на обучении: {model.score(X, y):.4f}%")