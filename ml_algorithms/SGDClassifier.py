#%%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming X_train_tfidf, Y_train, X_test_tfidf, Y_test are loaded

# Define the parameter grid for Random Search CV
param_dist = {
    'alpha': loguniform(1e-6, 1e-1),
    'eta0': [0.01, 0.1, 0.2, 0.5],
}

# Define custom scorer for roc_auc_score
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

# Train SGDClassifier model and perform Random Search CV
def SGDClassifier_train_random_search_cv(X_train, Y_train, X_test, Y_test):
    # Split the training data for cross-validation
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    # Initialize the classifier
    sgd_classifier = SGDClassifier(max_iter=1000, tol=1e-3)

    # Define RandomizedSearchCV
    random_search = RandomizedSearchCV(sgd_classifier, param_distributions=param_dist, n_iter=10, scoring=roc_auc_scorer, cv=3, random_state=0)

    # Perform RandomizedSearchCV
    random_search.fit(X_tr, Y_tr)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Use the best hyperparameters to train the final model
    best_sgd_classifier = SGDClassifier(max_iter=1000, tol=1e-3, alpha=best_params['alpha'], eta0=best_params['eta0'])
    best_sgd_classifier.fit(X_tr, Y_tr)

    # Predict probabilities for CV and training sets
    probs_cv = best_sgd_classifier.predict_proba(X_cv)[:, 1]
    probs_train = best_sgd_classifier.predict_proba(X_tr)[:, 1]

    # Calculate AUC score for CV and training sets
    auc_score_cv = roc_auc_score(Y_cv, probs_cv)
    auc_score_train = roc_auc_score(Y_tr, probs_train)

    # Calculate accuracy for CV and training sets
    threshold = 0.5
    binary_preds_cv = (probs_cv > threshold).astype(int)
    binary_preds_train = (probs_train > threshold).astype(int)
    accuracy_cv = accuracy_score(Y_cv, binary_preds_cv)
    accuracy_train = accuracy_score(Y_tr, binary_preds_train)

    print(f"AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
    print(f"AUC Score (Train): {auc_score_train}  Accuracy (Train): {accuracy_train}")

    # Confusion Matrix for CV set
    cm = confusion_matrix(Y_cv, binary_preds_cv)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Train')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Confusion Matrix for test set
    prob_test = best_sgd_classifier.predict_proba(X_test)[:, 1]
    binary_preds_test = (prob_test > threshold).astype(int)
    cm = confusion_matrix(Y_test, binary_preds_test)

    # Use ConfusionMatrixDisplay for visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title('Confusion Matrix Test')
    plt.show()

    print(Y_test, "\n")
    print(prob_test, "\n")
    print(binary_preds_test, "\n")

    # Calculate and print AUC score for test set
    auc_score = roc_auc_score(Y_test, prob_test)
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy for test set
    accuracy = accuracy_score(Y_test, binary_preds_test)
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy

# Call the function with your data
# SGDClassifier_train_random_search_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)
