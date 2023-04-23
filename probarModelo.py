#python probarModelo_aitziber.py -f iris.csv -m modelo.pkl -t Especie

# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
iFile = None
model = None
targetColumn = None
targetColumnName = None
algorithms = ['knn', 'decision tree']
algorithm = None
rescaleOption = None
imputeOption = None
excludedColumns = "tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,name,retweet_count,tweet_coord,tweet_created,tweet_location,user_timezone"
NLcolumns = ["text"]
NLtechnique = 'tfidf'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'t:m:f:ha:r:i:e:',['target=','model=','testFile=','h', 'algorithm=', 'rescale=', 'impute=', 'exclude='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)

    for opt,arg in options:
        if opt in ('-f', '--file'):
            iFile = arg
        elif opt in ('-m', '--model'):
            model = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)
        elif opt in ('-t', '--target'):
            targetColumnName = arg
        elif opt in ('-a', '--algorithm'):
            algorithms = ['knn', 'decision tree', 'logistic regression']
            algorithm = algorithms[int(arg)]
        elif opt in ('-r', '--rescale'):
            rescaleOption = arg
        elif opt in ('-i', '--impute'):
            imputeOption = arg
        elif opt in ('-e', '--exclude'):
            excludedColumns = arg
        elif opt in ('-c', '--technique'):
            NLtechnique = arg.lower()
            if NLtechnique != "bow" and NLtechnique != "tfidf":
                print("[!] The NL preprocess method should be \"bow\" or \"tfidf\"")
                sys.exit(1)
            

    if iFile == None or model == None:
        print('[!] Introduce input file o modelo')
        sys.exit(1)
    
    clf = pickle.load(open(model, 'rb'))

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    testX = pd.read_csv(iFile)
    preprocessor = Preprocessor()
    testX, target_map = preprocessor.preprocessDataset(testX, None, algorithm, excludedColumns, imputeOption, rescaleOption, NLcolumns)
    try:
        vocabulario = pickle.load(open(NLtechnique+".pkl", 'rb'))
        if NLtechnique == 'tfidf':
            tfidf = TfidfVectorizer(vocabulary=vocabulario)
            for columnaNL in NLcolumns:
                testX[columnaNL] = tfidf.fit_transform(testX[columnaNL])
        elif NLtechnique == 'bow':
            #TODO
            None
    except:
        None
    if targetColumnName != None:
        targetColumn = testX[targetColumnName]
        del testX[targetColumnName]

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')
    print(results_test)
    results_test.to_csv('predicciones.csv', index=False)
