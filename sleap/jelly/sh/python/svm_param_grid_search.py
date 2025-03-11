import sys
sys.path.append('../..')
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/python/')
from misprediction_classifier import load_dataset, load_20s_dataset, grid_search_with_resampling

import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, make_scorer, recall_score
from sklearn.svm import SVC

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ... existing code ...

# Set up logging at the start of your analysis
log_file = 'svm_param_grid_search_0310_3.txt'
sys.stdout = Logger(log_file)

def main():

    X_train, X_test, y_train, y_test, X, y, scaler = load_20s_dataset()

    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    class_weights_cand = [{0: 5, 1: 1}, {0: 10, 1: 1}]
    param_grid = dict(gamma=gamma_range, C=C_range, class_weight=class_weights_cand)

    best_params, best_score, best_model = grid_search_with_resampling(
        X_train, y_train, X_test, y_test,
        model=SVC(kernel='rbf'),
        param_grid=param_grid,
        minority_class=0,
        minor_to_major_ratio=0.2,
        dowmsample_majority_ratio=0.8
    )

    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")
    
    # Predict on test set and evaluate performance
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred))
    
    # Save the trained model and scaler
    model_save_path = 'svm_clf_20s_v3.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
    
sys.stdout = sys.__stdout__