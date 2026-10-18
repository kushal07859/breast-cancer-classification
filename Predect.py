import joblib
import kagglehub
import os
import pandas as pd

model=joblib.load("brest_cancer_model.pkl")

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
df = pd.read_csv(os.path.join(path,"data.csv"))

X=df.drop(columns=["diagnosis","id","Unnamed: 32"])

sample=X.iloc[[0]]

pre=model.predict(sample)

print(pre[0])
print(df["diagnosis"].iloc[0])