# This file is showing how model process training data 
# for pre-processing there is saperate file
# importing Required libraries
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug(' Model.py File execution started ')

# loading database with pandas library
df = pd.read_csv("./dataset/model.csv")
logging.debug(' Database Loaded ')

# model featuring
X = df.drop(['Price'],axis=1)
y = df['Price']

# Data Spliting For model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

# model fitting using LGBMRegressor
clf = lgb.LGBMRegressor()
clf.fit(X_train, y_train)

# Printing Accuracy
print(clf.score(X_test, y_test))

# pkl export & finish log
pickle.dump(clf, open("model.pkl", "wb"))
logging.debug(' Execution of Model.py is finished ')