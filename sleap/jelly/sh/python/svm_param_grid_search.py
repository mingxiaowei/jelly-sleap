import sys
sys.path.append('../..')
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/python/')
from misprediction_classifier import load_dataset, load_20s_dataset, grid_search_with_resampling

import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, make_scorer, recall_score
from sklearn.svm import SVC

def main():

    X_train, X_test, y_train, y_test, X, y, scaler = load_20s_dataset()

    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-2, 2, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)

    best_params, best_score, best_model = grid_search_with_resampling(
        X, y,
        model=SVC(kernel='rbf'),
        param_grid=param_grid,
        minority_class=0,
        minor_to_major_ratio=0.5,
        dowmsample_majority_ratio=0.8
    )

    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")
    
    # Predict on test set and evaluate performance
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save the trained model and scaler
    model_save_path = 'svm_clf_20s_v0.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()