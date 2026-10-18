# M = Malignant → Cancerous
# B = Benign → Non-cancerous

import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
df = pd.read_csv(os.path.join(path,"data.csv"))

df = df.drop(columns=["id", "Unnamed: 32"])

#labels=["radius_mean","perimeter_mean","area_mean","concavity_mean","concave points_mean",
        #"perimeter_se","area_se","perimeter_worst","area_worst","concave points_worst"]

y=df["diagnosis"]
X=df.drop(columns="diagnosis")

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
#MODEL SELECTION

models={
    "Logistic_Regression":Pipeline([
        ("Scaler",StandardScaler()),
        ("model",LogisticRegression(max_iter=1000))
    ]),

    "KNN":Pipeline([
        ("Scaler", StandardScaler()),
        ("model",KNeighborsClassifier())
    ]),

    "SVM":Pipeline([
        ("Scaler", StandardScaler()),
        ("model",SVC(random_state=42))
    ]),

    "Random Forest":RandomForestClassifier(
        n_estimators=100,random_state=42),

    "Gradient Boosting":GradientBoostingClassifier(random_state=42)
   
}

#cv 

cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "accuracy": "accuracy",
    "malignant_recall": make_scorer(
        recall_score, pos_label="M"
    ),
    "malignant_f1": make_scorer(
        f1_score, pos_label="M"
    ),
    "roc_auc": "roc_auc"
}

results=[]

for name, model in models.items():
    scores=cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "model":name,
        "Accuracy":scores["test_accuracy"].mean(),
        "Malignant Recall":scores["test_malignant_recall"].mean(),
        "malignant F1": scores["test_malignant_f1"].mean(),
        "Roc-Auc":scores["test_roc_auc"].mean()
    }) 

results_df = pd.DataFrame(results)

print(results_df.round(4).to_string(index=False))

lr_pipeline=Pipeline([
    ("Scaler",StandardScaler()),
    ("LR",LogisticRegression(max_iter=5000))
])

param_grid={
    "LR__C":[0.01,0.1,1,10,100],
    "LR__solver":["liblinear"],
     "LR__l1_ratio": [0]     #0 = l2 and 1 = l1
}

lr_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1
)

lr_search.fit(X_train,y_train)

best_model=lr_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred,labels=["B","M"]))

cm=confusion_matrix(y_test,y_pred,labels=["B","M"])

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["BENIGN (B)", "MALIGNANT (M)"],
    yticklabels=["BENIGN (B)", "MALIGNANT (M)"],
    cbar=False
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

y_proba=best_model.predict_proba(X_test)
m_index=list(best_model.classes_).index("M")
y_proba_m = y_proba[:,m_index]

y_test_m= (y_test == "M").astype(int)

fpr, tpr, thresholds = roc_curve(y_test_m, y_proba_m)

auc = roc_auc_score(y_test_m, y_proba_m)

print("ROC-AUC",auc)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"Logistic Regression (Auc = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

import joblib
joblib.dump(best_model, "brest_cancer_model.pkl")