# %%
import pandas as pd
from sklearn import pipeline
from sklearn import tree
print("Importado")
# %%

# %% 
df_celula = pd.read_csv("../data/celulas.csv", sep=";")

df_celula
# %%

# %%

# Como criar variáveis Dummys
df_dummie = pd.get_dummies(df_celula[["cor", "membrana"]])

df_tentação = pd.concat([df_celula, df_dummie], axis=1)

del df_tentação["cor"]
del df_tentação["membrana"]

df_tentação

# %%
features = ["nucleos",
            "caudas",
            "cor_Clara",
            "cor_Escura",
            "membrana_Fina",
            "membrana_Grossa"]

target = "classe"
clf_tree = tree.DecisionTreeClassifier()

clf_tree.fit(df_tentação[features], df_tentação[target])
# %%

# Salvando

model = pd.Series({"model":clf_tree,
        "features": features,
        "target": target
})

model.to_pickle("../models/celulas_tree_dummies.pkl")
model
# %%
