import sys
sys.path.append('../..')
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/python/')
from misprediction_classifier import load_dataset

import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, make_scorer, recall_score
from sklearn.svm import SVC

def main():
    # Create a custom scorer that focuses on recall for class 0
    recall_0_scorer = make_scorer(recall_score, pos_label=0)

    dist_thres = 6
    X_train, X_test, y_train, y_test, X_scaled, y = load_dataset(dist_thres=dist_thres)

    C_range = np.logspace(-2, 4, 7)
    gamma_range = np.logspace(-2, 4, 7)
    kernel_range = ['linear', 'rbf', 'sigmoid']
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernel_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring=recall_0_scorer, verbose=2)
    grid.fit(X_scaled, y)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

    # Predict on test set
    y_pred = grid.predict(X_test)

    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save the trained model and scaler
    model_save_path = f'svm_clf_thres_{dist_thres}.joblib'
    joblib.dump(grid.best_estimator_, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()