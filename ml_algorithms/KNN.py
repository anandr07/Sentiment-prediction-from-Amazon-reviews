#%%[markdown]
# Here I have implemented KNN algorithm using sklearn using BOW, tf-idf, Average word2vec and tf-idf word2vec

# %%
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split 

#%%
#%%[markdown]
# Here I have implemented KNN algorithm using sklearn using BOW, tf-idf, Average word2vec and tf-idf word2vec

# %%
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer

#%%
#find knn to simple cross validation with Brute Force and KD-Tree
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def KNN_train_simple_cv(X_train, Y_train):
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    k = []
    pred_cv_auc = []
    pred_train_auc = []
    pred_cv_accuracy = []
    pred_train_accuracy = []
    max_roc_auc = -1
    best_k_auc = 0
    best_accuracy = 0
    for i in range(1, 24, 2):
        knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute',n_jobs=-1)
        knn.fit(X_tr, Y_tr)
        probs_cv = knn.predict_proba(X_cv)[:, 1]
        probs_train = knn.predict_proba(X_tr)[:, 1]

        auc_score_cv = roc_auc_score(Y_cv, probs_cv)  # find AUC score
        auc_score_train = roc_auc_score(Y_tr, probs_train)

        # Calculate accuracy
        threshold = 0.5
        binary_preds_cv = (probs_cv > threshold).astype(int)
        binary_preds_train = (probs_train > threshold).astype(int)
        accuracy_cv = accuracy_score(Y_cv, binary_preds_cv)
        accuracy_train = accuracy_score(Y_tr, binary_preds_train)

        print(f"{i} - AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
        pred_cv_auc.append(auc_score_cv)
        pred_train_auc.append(auc_score_train)
        pred_cv_accuracy.append(accuracy_cv)
        pred_train_accuracy.append(accuracy_train)
        k.append(i)

        if max_roc_auc < auc_score_cv:
            max_roc_auc = auc_score_cv
            best_k_auc = i
        if best_accuracy < accuracy_cv:
            best_accuracy = accuracy_cv

    print(f"Best k-value based on AUC: {best_k_auc}")
    print(f"Best accuracy: {best_accuracy}")

    # Plotting k vs AUC Score
    plt.plot(k, pred_cv_auc, 'r-', label='CV AUC Score')
    plt.plot(k, pred_train_auc, 'g-', label='Train AUC Score')
    plt.legend(loc='upper right')
    plt.title("k vs AUC Score")
    plt.ylabel('AUC Score')
    plt.xlabel('k')
    plt.show()

    # Plotting k vs Accuracy
    plt.plot(k, pred_cv_accuracy, 'b-', label='CV Accuracy')
    plt.plot(k, pred_train_accuracy, 'c-', label='Train Accuracy')
    plt.legend(loc='upper right')
    plt.title("k vs Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.show()

    # Confusion Matrix for best k based on accuracy
    knn = KNeighborsClassifier(n_neighbors=best_k_auc, algorithm='brute')
    knn.fit(X_train, Y_train)
    prob_cv = knn.predict_proba(X_cv)[:, 1]
    binary_preds_cv = (prob_cv > threshold).astype(int)
    cm = confusion_matrix(Y_cv, binary_preds_cv)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Train')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display values inside the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.show()

    return knn

#%%

def KNN_test(trained_KNN_Model, X_test, Y_test):
    threshold = 0.5
    prob_test = trained_KNN_Model.predict_proba(X_test)[:, 1]
    binary_preds_test = (prob_test > threshold).astype(int)
    cm = confusion_matrix(Y_test, binary_preds_test)

    # Display values inside the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Test')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Calculate and print AUC score
    auc_score = roc_auc_score(Y_test, prob_test)
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy
    accuracy = accuracy_score(Y_test, binary_preds_test)
    print(f"Accuracy (Test): {accuracy}")

    plt.show()  

    return auc_score, accuracy
