
# %%
import pandas as pd
from sklearn import pipeline
from sklearn import tree
from feature_engine.encoding import OneHotEncoder
# %%
df_celula = pd.read_csv("../data/celulas.csv", sep=";")
df_celula