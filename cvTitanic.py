import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_model(n_hidden = 1, n_neurons=30, input_shape=[6]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='elu'))
        keras.layers.Dropout(0.1),
    model.add(keras.layers.Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = KerasClassifier(build_model)

titanic = pd.read_csv("data/train.csv")

sexes = titanic[["Sex"]]
hotEncoder = OrdinalEncoder()
sexes_hot = hotEncoder.fit_transform(sexes)
titanic["Sex_new"] = sexes_hot

drop = ["PassengerId", "Name", "Sex", "Cabin", "Ticket", "Embarked"]
titanic_droped = titanic.drop(labels=drop, axis = 1)

titanic_labels = titanic_droped["Survived"].values
titanic_data = titanic_droped.drop(labels="Survived", axis = 1)

imputer = SimpleImputer(strategy="mean")
titanic_data_filled_array = imputer.fit_transform(titanic_data)

titanic_filled_data = pd.DataFrame(
    titanic_data_filled_array,
    columns = titanic_data.columns,
    index = titanic_data.index
)

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

titanic_data = num_pipeline.fit_transform(titanic_filled_data)

X_train, X_valid, y_train, y_valid = train_test_split(titanic_data, titanic_labels, test_size=0.1)

param_distribs = {
    'n_hidden'  : [(1),(2),(3),(4),(5),(6)],
    'n_neurons' : [(10),(20),(30),(40),(50),(60)]
}

rnd_search_cv = RandomizedSearchCV(model, param_distribs, n_iter=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)])

print("Best parameters set found on development set:")
print()
print(rnd_search_cv.best_params_)
print()
print("Grid scores on development set:")
print()
means = rnd_search_cv.cv_results_['mean_test_score']
stds = rnd_search_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rnd_search_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
# model.fit(titanic_data,titanic_labels,epochs=10, verbose=0)

# model.fit(X_train, y_train, epochs=100, verbose=1,
#                   validation_data=(X_valid, y_valid),
#                   callbacks=[keras.callbacks.EarlyStopping(patience=10)])
