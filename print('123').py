"""
gradient_descent_wape.py - Краткая библиотека градиентного спуска с WAPE
"""

import numpy as np


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100


class LinearRegressionGD:
    """Линейная регрессия с градиентным спуском и оптимизацией по WAPE"""
    
    def __init__(self, lr=0.01, epochs=1000, batch_size=None, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y, verbose=False):
        X, y = np.array(X), np.array(y).flatten()
        n, m = X.shape
        
        # Инициализация
        self.weights = np.zeros(m)
        self.bias = 0.0
        batch_size = self.batch_size or n
        
        prev_loss = np.inf
        
        for epoch in range(self.epochs):
            # Перемешивание и разбиение на батчи
            idx = np.random.permutation(n)
            X_shuffled, y_shuffled = X[idx], y[idx]
            
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
            
            # Мониторинг
            y_pred_all = X @ self.weights + self.bias
            loss = wape(y, y_pred_all)
            self.loss_history.append(loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, WAPE: {loss:.4f}%")
            
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"Сошлось на эпохе {epoch+1}")
                break
            prev_loss = loss
        
        return self
    
    def predict(self, X):
        return np.array(X) @ self.weights + self.bias
    
    def score(self, X, y):
        return wape(y, self.predict(X))


# Пример использования
if __name__ == "__main__":
    # Данные
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3*X[:, 0] + 2*X[:, 1] + 1 + np.random.randn(100) * 0.1
    
    # Обучение
    model = LinearRegressionGD(lr=0.05, epochs=500, batch_size=20)
    model.fit(X, y, verbose=True)
    
    # Результаты
    print(f"\nВеса: {model.weights}")
    print(f"Смещение: {model.bias:.4f}")
    print(f"WAPE на обучении: {model.score(X, y):.4f}%")