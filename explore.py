import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
# from sklearn.preprocessing import OrdinalEncoder

titanic = pd.read_csv("data/train.csv")
# print(titanic.head())
# print(titanic.info())
# print(titanic.describe())

# Encode Sex into Male and Female cats
sexes = titanic[["Sex"]]
hotEncoder = OneHotEncoder()
sexes_hot = hotEncoder.fit_transform(sexes)
sexes_hot = sexes_hot.toarray()
titanic["Female"] = sexes_hot[:,0]
titanic["Male"] = sexes_hot[:,1]

# Drop data that is not needed
drop = ["PassengerId", "Name", "Sex", "Cabin", "Ticket", "Embarked"]
titanic_droped = titanic.drop(labels=drop, axis = 1)
# print(titanic_droped.info())
# print(titanic_droped.describe())

imputer = SimpleImputer(strategy="mean")
# imputer = SimpleImputer(strategy="median")
titanic_filled_array = imputer.fit_transform(titanic_droped)
titanic_filled = pd.DataFrame(
    titanic_filled_array,
    columns = titanic_droped.columns,
    index = titanic_droped.index
)

# print(titanic_filled.info())
# print(titanic_filled.describe())

# titanic_filled.hist(bins=20)
# plt.show()

# corr_matrix = titanic_filled.corr()
# corr_relationships = corr_matrix["Survived"].sort_values(ascending = False)
# print(corr_relationships)
#
# pd.plotting.scatter_matrix(titanic_filled[corr_relationships.index])
# plt.show()

titanic_labels = titanic_filled["Survived"]
titanic_filled_data = titanic_filled.drop(labels="Survived", axis = 1)

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

titanic_data = num_pipeline.fit_transform(titanic_filled_data)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(titanic_data)
PCAX = pca.transform(titanic_data)
print(pca.explained_variance_ratio_.sum())
plt.scatter(PCAX[:, 0], PCAX[:, 1], c = titanic_labels)
plt.show()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
tsne_results = tsne.fit_transform(titanic_data)
plt.scatter(PCAX[:, 0], PCAX[:, 1], c=titanic_labels)
plt.show()
