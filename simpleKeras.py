import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

TRAIN = False

titanic = pd.read_csv("data/train.csv")
titanicTEST = pd.read_csv("data/test.csv")

# titanic = titanic.sample(frac=1).reset_index(drop=True)

# Encode Sex into Male and Female cats
sexes = titanic[["Sex"]]
hotEncoder = OneHotEncoder()
sexes_hot = hotEncoder.fit_transform(sexes)
sexes_hot = sexes_hot.toarray()
titanic["Female"] = sexes_hot[:,0]
titanic["Male"] = sexes_hot[:,1]

# Encode Sex into Male and Female cats
sexesTEST = titanicTEST[["Sex"]]
hotEncoderTEST = OneHotEncoder()
sexes_hotTEST = hotEncoderTEST.fit_transform(sexesTEST)
sexes_hotTEST = sexes_hotTEST.toarray()
titanicTEST["Female"] = sexes_hotTEST[:,0]
titanicTEST["Male"] = sexes_hotTEST[:,1]

# Drop data that is not needed
drop = ["PassengerId", "Name", "Sex", "Cabin", "Ticket", "Embarked"]
titanic_droped = titanic.drop(labels=drop, axis = 1)
titanic_dropedTEST = titanicTEST.drop(labels=drop, axis = 1)

titanic_labels = titanic_droped["Survived"].values
titanic_data = titanic_droped.drop(labels="Survived", axis = 1)

# titanic_labelsTEST = titanic_dropedTEST["Survived"].values
# titanic_dataTEST = titanic_dropedTEST.drop(labels="Survived", axis = 1)

imputer = SimpleImputer(strategy="mean")
# imputer = SimpleImputer(strategy="median")
titanic_data_filled_array = imputer.fit_transform(titanic_data)
titanic_data_filled_arrayTEST = imputer.transform(titanic_dropedTEST)
titanic_filled_data = pd.DataFrame(
    titanic_data_filled_array,
    columns = titanic_data.columns,
    index = titanic_data.index
)

titanic_filledTEST = pd.DataFrame(
    titanic_data_filled_arrayTEST,
    columns = titanic_dropedTEST.columns,
    index = titanic_dropedTEST.index
)

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

titanic_data = num_pipeline.fit_transform(titanic_filled_data)
titanic_dataTEST = num_pipeline.transform(titanic_filledTEST)

if TRAIN:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvscores = []
    accscores = []
    index = 1
    for train, test in kfold.split(titanic_data, titanic_labels):
        # create model
        model = keras.models.Sequential([
            keras.layers.InputLayer(7),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(titanic_data[train], titanic_labels[train], epochs=80, verbose=0)
        scores = model.evaluate(titanic_data[test], titanic_labels[test], verbose=0)
        print(str(index) + " - Training: %.2f%% CV: %.2f%%" % (history.history["accuracy"][-1] * 100, scores[1] * 100))
        index += 1
        cvscores.append(scores[1] * 100)
        accscores.append(history.history["accuracy"][-1] * 100)

    print("\nTraining   : %.2f%% (+/- %.2f%%)" % (np.mean(accscores), np.std(accscores)))
    print("Validation : %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Bias       : %.2f%%" % (95-np.mean(accscores)))
    print("Variance   : %.2f%%" % (abs(np.mean(accscores)-np.mean(cvscores))))

model = keras.models.Sequential([
    keras.layers.InputLayer(7),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training Final Model...")
model.fit(titanic_data, titanic_labels, epochs=80, verbose=0)
print("Predictions...")
preds = model.predict(titanic_dataTEST)
preds = preds > 0.5

startID = 892
index = 0
with open('data/results.csv', 'w') as results:
    results.write('PassengerId,Survived\n')
    for index in range(418):
        results.write(str(startID) + "," + str(int(preds[index]))+ "\n")
        startID += 1
        pass


# model = keras.models.Sequential([
#     keras.layers.InputLayer(7),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10, activation='relu'),
#     keras.layers.Dense(5, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# history = model.fit(titanic_data, titanic_labels, epochs=100, validation_split=0.1)
#
# pd.DataFrame(history.history).plot()
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()
