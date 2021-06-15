
# %%
import pandas as pd
from sklearn import pipeline
from sklearn import tree
from feature_engine.encoding import OneHotEncoder
# %%
df_celula = pd.read_csv("../data/celulas.csv", sep=";")
df_celula
# %%
features = ["nucleos", "caudas", "cor", "membrana"]
target = "classe"

onehot = OneHotEncoder(variables=["cor", "membrana"])

clf_tree = tree.DecisionTreeClassifier()

model_pipeline = pipeline.Pipeline(steps= [("Onehot", onehot),
                                    ("Tree", clf_tree)])

model_pipeline.fit(df_celula[features], df_celula[target])
# %%
model = pd.Series( {
    "model": model_pipeline,
    "features": features,
    "target":target
})

model.to_pickle("../models/celulas_tree_pipeline.pkl")
# %%
