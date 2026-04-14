import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

Input_Dir = r'C:\Users\ABHAY SINHA\Parking_Lot_Project\clf-data\clf-data'
Categories = ['empty', 'not_empty']

Data = []
Labels = []

for Catg_idx, Category in enumerate(Categories):
    for File in os.listdir(os.path.join(Input_Dir, Category)):
        Image_Path = os.path.join(Input_Dir, Category, File)
        Image = imread(Image_Path)
        Image = resize(Image, (15,15))
        Data.append(Image.flatten())
        Labels.append(Catg_idx)

Data = np.asarray(Data)
Labels = np.asarray(Labels)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(Data, Labels, test_size=0.2, shuffle=True, stratify=Labels)

Classifier = SVC()

Parmaters = ({'gamma' : [0.01, 0.001, 0.0001], 'C':[1,10,100,1000]})

Grid_Search = GridSearchCV(Classifier, Parmaters)
Grid_Search.fit(X_Test,Y_Test)

Best_Estimator = Grid_Search.best_estimator_
Y_Predication = Best_Estimator.predict(X_Test)

Score = accuracy_score(Y_Predication, Y_Test)
print(Score)

pickle.dump(Best_Estimator, open('Model.p', 'wb'))

