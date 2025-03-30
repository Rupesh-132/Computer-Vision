## Flow of the project
# Prepare dataset --> train/test ---> train classifier -->  # test performance

import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


input_dir = "Day-4-Parking-Lot-Classification\data\clf-data\clf-data"
categories = ["empty","not_empty"]

data = []
labels = []

for category_index, category in enumerate(categories): 
    for file in os.listdir(os.path.join(input_dir,category)):
        
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img,(15,15))  # resizing the image after reading it
        data.append(img.flatten()) # flattening the image into 1D array
        labels.append(category_index) # appending the category index into the 
        
    
# Casting each of the list into the numpy array
data = np.asarray(data)
labels = np.asarray(labels)

# train/test split of the prepared data
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2, shuffle= True, stratify= labels) # stratify ensures the classes are properly distributed into train and test properly

classifier = SVC()

parameters = [
    {
    "gamma" : [0.01,0.001,0.0001],
    "C" : [1,10,100,1000]
    
        }
    ]


grid_search = GridSearchCV(classifier,parameters)

grid_search.fit(X_train,y_train)


# test performance
best_estimator = grid_search.best_estimator_
print(best_estimator)

y_pred = best_estimator.predict(X_test)

score = accuracy_score(y_test,y_pred)

print("The accuracy of the trained classifier is:",score)







