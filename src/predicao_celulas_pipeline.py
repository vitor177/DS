
# %%
import pandas as pd

model = pd.read_pickle("../models/celulas_tree_pipeline.pkl")

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


pred = model["model"].predict(data[model["features"]])[0]

print(f"A célula é do tipo: {pred}")


# %%
