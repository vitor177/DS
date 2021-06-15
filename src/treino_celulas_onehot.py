
# %%
import pandas as pd
from sklearn import pipeline
from sklearn import tree
from feature_engine.encoding import OneHotEncoder

df_celula = pd.read_csv("../data/celulas.csv", sep=";")


# %%
features = ["nucleos", "caudas", "cor", "membrana"]
onehot = OneHotEncoder(variables=["cor", "membrana"])
onehot.fit(df_celula[features])
df_fit = onehot.transform(df_celula[features])
# %%

target = "classe"
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(df_fit, df_celula[target])

# %%
model = pd.Series({"model":clf_tree,
        "onehot": onehot,
        "features": features,
        "target": target
})

model.to_pickle("../models/celulas_tree_dummies_onehot.pkl")
model
