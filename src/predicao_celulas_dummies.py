
# %%
import pandas as pd

model = pd.read_pickle("../models/celulas_tree_dummies.pkl")

model
# %%

nucleos = int(input("Digite a quantidade de núcleos"))
caudas = int(input("Digite a quantidade de caudas"))
cor = input("Digite a cor")
membrana = input("Digite a membrana")

data = pd.DataFrame({"nucleos": [nucleos],
                        "caudas": [caudas],
                        "cor": [cor],
                        "membrana": [membrana]})

data
# %%

df_new = pd.get_dummies( data[["cor", "membrana"]])

df_new
# %%
df_full = pd.concat([data, df_new], axis=1)

df_full
# %%
for f in model['features']:
    if f not in df_full.columns:
        df_full[f] = 0


df_full
# %%
df_full = df_full[model['features']]

df_full

# model["model"].predict(data)
# %%

pred = model["model"].predict(df_full)[0]

pred
# %%
print(f"A célula é do tipo: {pred}")
# %%
