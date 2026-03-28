import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================
# Chargement des données
# =============================

file_path = "IRIS.xlsx"
df = pd.read_excel(file_path, sheet_name="Feuil1").dropna()

print("Structure du dataset")
print(df.info())

# Variables quantitatives
X = df[["SE_L","SE_W","PE_L","PE_W"]]

# =============================
# Statistiques descriptives
# =============================

print("\nStatistiques globales")
print(X.describe())

print("\nMoyennes")
print(X.mean())

print("\nÉcart-types")
print(X.std())

print("\nStatistiques par espèce")
print(df.groupby("Espece").describe())

# =============================
# Graphiques
# =============================

sns.histplot(df["PE_L"], kde=True)
plt.title("Histogramme + densité PE_L")
plt.show()

sns.histplot(data=df, x="PE_L", hue="Espece", kde=True)
plt.show()

sns.boxplot(x="Espece", y="PE_L", data=df)
plt.title("Boxplot PE_L par espèce")
plt.show()

sns.scatterplot(x="PE_L", y="PE_W", hue="Espece", data=df)
plt.title("PE_L vs PE_W")
plt.show()

sns.pairplot(df, hue="Espece")
plt.show()

# =============================
# Standardisation
# =============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# Matrice de corrélation
# =============================

corr_matrix = X.corr()

print("\nMatrice de corrélation")
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show()

# =============================
# Valeurs propres et vecteurs propres
# =============================

eig_values, eig_vectors = np.linalg.eig(corr_matrix)

print("\nValeurs propres")
print(eig_values)

print("\nVecteurs propres")
print(eig_vectors)

# Variance expliquée
variance_explained = eig_values / np.sum(eig_values) * 100

print("\nVariance expliquée (%)")
for i,v in enumerate(variance_explained):
    print(f"Axe {i+1} : {v:.2f}%")

# Dimension du sous-espace de projection
dimension = np.sum(variance_explained > 5)
print("\nDimension du sous-espace factoriel :", dimension)

# =============================
# ACP avec sklearn
# =============================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nVariance expliquée sklearn (%)")
print(pca.explained_variance_ratio_ * 100)

# =============================
# Coordonnées d’un individu
# =============================

if "Num" in df.columns:
    idx = df[df["Num"] == "i027"].index

    if len(idx) > 0:
        i = idx[0]
        print("\nCoordonnées i027 sur PC1 PC2")
        print(X_pca[i,0:2])

# =============================
# Cos² (qualité de représentation)
# =============================

# cos² = carré des coordonnées / somme des carrés

cos2 = X_pca**2
cos2 = cos2 / np.sum(X_pca**2, axis=1).reshape(-1,1)

print("\nQualité de représentation i027")
if len(idx) > 0:
    print(cos2[i,0:2])

# =============================
# Contribution des variables
# =============================

contrib_variables = np.abs(pca.components_[0])
max_var_index = np.argmax(contrib_variables)

print("\nVariable qui contribue le plus au premier axe :")
print(X.columns[max_var_index])

# =============================
# Cercle de corrélation
# =============================

plt.figure(figsize=(6,6))

for i,col in enumerate(X.columns):
    plt.arrow(0,0,
              pca.components_[0,i],
              pca.components_[1,i],
              head_width=0.02)

    plt.text(pca.components_[0,i],
             pca.components_[1,i],
             col)

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.axhline(0)
plt.axvline(0)
plt.title("Cercle de corrélation")
plt.gca().set_aspect('equal')
plt.show()

# =============================
# Projection des individus
# =============================

plt.figure()

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df["Espece"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection des individus")
plt.show()

# =============================
# Fleur la plus contributive
# =============================

if "Num" in df.columns:
    contribution = np.sum(np.abs(X_pca[:,0]))

    max_index = np.argmax(np.abs(X_pca[:,0]))

    print("\nFleur qui contribue le plus à l’axe 1 :")
    print(df.iloc[max_index]["Num"])

