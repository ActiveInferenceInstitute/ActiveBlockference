import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression as mi

import argparse
parser.add_argument('viz', help='Description for bar argument', required=True)
parser.add_argument('X', type=argparse.FileType('r'), required=False)
args = vars(parser.parse_args())

condition = args['viz'] == 'v'

# X : paramter at step n, y : same paramter at step n+1
def calculate_mi(X, y):   
    ## Preprocess and int encode if not already
    df = pd.read(X)
    X.select_dtypes('int')

    for colname in X.select_dtypes('object'): 
        X[colname], uniques = X[colname].factorize()
        discrete_features = X.dtypes

    mi_scores = mi(X,y,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores,name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores


if __name__ == "__main__":
    if condtion:
        import seaborn as sns
        import matplotlib.pyplot as plt
        calculate_mi(X, y, discrete_features)
        #sns.lmplot(x=.. , y=.. , hue=.. , data=..)
