import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('UNSW_NB15_training-set.csv')

# Supprimer la dernière colonne
df = df.iloc[:, :-1]

# Sauvegarder le fichier sans la dernière colonne
df.to_csv('ton_fichier_sans_derniere_colonne.csv', index=False)
