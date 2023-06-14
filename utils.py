import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def mean_adjacent(arr: np.ndarray, window_size: int = 2) -> np.ndarray:
    if len(arr) < 2:
        return arr.squeeze()
    # Criando uma janela deslizante
    windowed_view = sliding_window_view(arr, window_size)
    
    # Calculate the mean of each subarray of adjacent numbers
    means = np.mean(windowed_view, axis=1)
    
    return means

def cost_function(l_labels, r_labels, classes):
    m_left, m_right = len(l_labels), len(r_labels)
    m = m_left + m_right

    g_left = gini_impurity(l_labels)
    g_right = gini_impurity(r_labels)

    return g_left*(m_left/m) + g_right*(m_right/m)

def proportion(class_, data):
    m = len(data)
    if len(data) == 0:
        return 0
    return np.count_nonzero(data == class_) / m

def gini(p_classes):
    return np.sum(p_classes**2)

def proportions(classes, y):
    return np.apply_along_axis(lambda class_: proportion(class_, y), 
                                    arr=classes.reshape(-1, 1), 
                                    axis=1)

def gini_impurity(y):
    classes = np.unique(y)
    p_classes = proportions(classes, y)
    return 1 - gini(p_classes)

def get_majority(classes, y):
    return np.argmax(proportions(classes , y))