from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import sparse

# Train Multinomial Naive Bayes model and get AUC score and accuracy
def NaiveBayes_train_simple_cv(X_train, Y_train, X_test, Y_test):
    # Split the training data for cross-validation
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    # Initialize the classifier
    nb = MultinomialNB()

    # Train the classifier
    nb.fit(X_tr, Y_tr)

    # Predict probabilities for CV and training sets
    probs_cv = nb.predict_proba(X_cv)[:, 1]
    probs_train = nb.predict_proba(X_tr)[:, 1]

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
    prob_test = nb.predict_proba(X_test)[:, 1]
    binary_preds_test = (prob_test > threshold).astype(int)
    cm = confusion_matrix(Y_test, binary_preds_test)

    # Use ConfusionMatrixDisplay for visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title('Confusion Matrix Test')
    plt.show()  

    # Calculate and print AUC score for test set
    auc_score = roc_auc_score(Y_test, prob_test)
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy for test set
    accuracy = accuracy_score(Y_test, binary_preds_test)
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy
