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

#%%
def KNN_train_n_5_fold_cv(X_train,Y_train):
    k = []
    best_k_auc = 0
    best_k_accuracy = 0
    pred_cv_accuracy = []
    pred_cv_auc_roc = []
    pred_train_accuracy = []
    pred_train_auc_roc = []
    best_roc_auc = 0
    best_accuracy = 0 
    for i in range(1,24,2):
        k.append(i)
        knn = KNeighborsClassifier(n_neighbors=i,algorithm='kd_tree')
        knn.fit(X_train,Y_train)

        cv = KFold(n_splits=5)
        cv_accuracy = cross_val_score(knn, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        cv_auc_roc = cross_val_score(knn, X_train, Y_train, scoring='roc_auc', cv=cv, n_jobs=-1)

        pred_cv_accuracy.append(np.mean(cv_accuracy))
        pred_cv_auc_roc.append(np.mean(cv_auc_roc))

        prob = knn.predict_proba(X_train)
        prob  = prob[:, 1]
        
        threshold = 0.5
        binary_preds = (prob > threshold).astype(int)

        auc_score_train = roc_auc_score(Y_train,binary_preds)
        accuracy_train = accuracy_score(Y_train,binary_preds)

        print(f"k value : {i}  CV Accuracy : {np.mean(cv_accuracy)}  CV AUC-ROC Score : {np.mean(cv_auc_roc)}")
        pred_train_auc_roc.append(auc_score_train)
        pred_train_accuracy.append(accuracy_train)

        if(best_roc_auc < np.mean(cv_auc_roc)):
            best_roc_auc = np.mean(cv_auc_roc)
            best_k_auc = i
        if(best_accuracy < np.mean(cv_accuracy)):
            best_accuracy = np.mean(cv_accuracy)
            best_k_accuracy = i

    print(f"k-value when best Accuracy was achieved is : {best_k_auc} ")
    print(f"k-value when best AUC was achieved is : {best_k_accuracy} ")

    # Plotting k vs accuracy
    plt.plot(k, pred_cv_accuracy, label='CV Accuracy')
    plt.plot(k, pred_train_accuracy, label='Train Accuracy')
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('k vs Accuracy')
    plt.show()

    # Plotting k vs AUC-ROC Score
    plt.plot(k, pred_cv_auc_roc, label='CV AUC-ROC Score')
    plt.plot(k, pred_train_auc_roc, label='Train AUC-ROC Score')
    plt.xlabel('k value')
    plt.ylabel('AUC-ROC Score')
    plt.legend()
    plt.title('k vs AUC-ROC Score')
    plt.show()

    # Plotting Confusion Matrix
    knn = KNeighborsClassifier(n_neighbors=best_k_accuracy, algorithm='kd_tree')
    knn.fit(X_train, Y_train)
    prob = knn.predict_proba(X_train)
    prob = prob[:, 1]
    y_pred = (prob > 0.5).astype(int)
    cm = confusion_matrix(Y_train, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plotting ROC Curve
    fpr, tpr, _ = roc_curve(Y_train, binary_preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#%%
