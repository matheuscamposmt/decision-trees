import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def mean_adjacent(arr: np.ndarray, window_size: int = 2) -> np.ndarray:
    if len(arr) < 2:
        return arr
    # Criando uma janela deslizante
    windowed_view = sliding_window_view(arr, window_size)
    
    # Calculate the mean of each subarray of adjacent numbers
    means = np.mean(windowed_view, axis=1)
    
    return means

def impurity_function(l_labels, r_labels) -> float:
    m_left, m_right = len(l_labels), len(r_labels)
    m = m_left + m_right

    g_left = gini_impurity(l_labels)
    g_right = gini_impurity(r_labels)

    return g_left*(m_left/m) + g_right*(m_right/m)

def loss_function(l_labels, r_labels):
    m_left, m_right = len(l_labels), len(r_labels)
    m = m_left + m_right

    s_left = sse(l_labels)
    s_right = sse(r_labels)

    return s_left*(m_left/m) + s_right*(m_right/m)

def sse(y) -> float:
    return np.sum((y - get_mean(y))**2)

def get_mean(y) -> float:
    return np.mean(y)

def proportion(class_, data) -> float:
    m = len(data)
    if len(data) == 0:
        return 0
    return np.count_nonzero(data == class_) / m

def proportions(classes, y):
        return np.apply_along_axis(lambda class_: proportion(class_, y), 
                                        arr=classes.reshape(-1, 1), 
                                        axis=1)

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0
    
    classes = np.unique(y)
    p_classes = proportions(classes, y)
    
    return 1 - np.sum(p_classes**2)

def get_majority_class(y: np.ndarray) -> int:
    classes = np.unique(y).astype(np.int64)
    return classes[np.argmax(proportions(classes, y))]