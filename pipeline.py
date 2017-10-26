import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

import pickle

features = pd.read_csv('final_data.csv')
target = pd.read_csv('target.csv').loc[:, '0']

features = features.drop('Unnamed: 0', axis=1)

def asdasd():
    # NOTE: Make sure that the class is labeled 'target' in the data file
    #tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
    #features = tpot_data.drop('target', axis=1).values
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, target, random_state=42)

    # Score on the training set was:0.7130827067669173
    exported_pipeline = make_pipeline(
        RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.55, n_estimators=100), step=0.35000000000000003),
        StackingEstimator(estimator=GaussianNB()),
        ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=6, min_samples_split=4, n_estimators=100)
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    with open('pickled_pipeline.pkl', 'wb') as fid:
        pickle.dump(exported_pipeline, fid)

asdasd()