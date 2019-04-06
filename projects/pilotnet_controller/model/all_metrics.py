import numpy as np

def rmse(outputs, targets):
    residuals = (outputs - targets)
    return np.sqrt(np.mean(residuals**2))
    
# maintain all metrics required in this dictionary- these are used in the
# training and evaluation loops
metrics = {
    'rmse': rmse
}
