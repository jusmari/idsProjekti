import json
import pandas as import pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split as tts

import urllib.request

data = pd.read_csv('sampledata.csv')



#X_train, X_test, y_train, y_test = tts()