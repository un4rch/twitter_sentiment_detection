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
from sklearn.feature_extraction.text import CountVectorizer

iFile = None
model = None
targetColumn = None
targetColumnName = None
algorithms = ['knn', 'decision tree']
algorithm = None
rescaleOption = None
imputeOption = 'MODE'
excludedColumns = "tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,name,retweet_count,tweet_coord,tweet_created,tweet_location,user_timezone"
NLcolumns = ["text"]
NLtechnique = None
NLvectorizer = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'t:m:f:ha:r:i:e:v:',['target=','model=','testFile=','h', 'algorithm=', 'rescale=', 'impute=', 'exclude=', 'vectorize='])
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
        elif opt in ('-v', '--vectorize'):
            NLtechnique = arg.lower()
            if NLtechnique != "bow" and NLtechnique != "tfidf":
                print("[!] The NL preprocess method should be \"bow\" or \"tfidf\"")
                sys.exit(1)
            

    if iFile == None or model == None:
        print('[!] Introduce input file o modelo')
        sys.exit(1)
    if NLcolumns != None and NLtechnique == None:
        print('[!] Introduce la tecnica de vectorizado tfidf o bow')
        sys.exit(1)
    
    clf = pickle.load(open(model, 'rb'))

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    testX = pd.read_csv(iFile)
    preprocessor = Preprocessor()
    testX, target_map = preprocessor.preprocessDataset(testX, targetColumnName, algorithm, excludedColumns, imputeOption, rescaleOption, NLcolumns, NLtechnique, "test", "on")
    target_map_reves = {}
    for key in target_map:
        value = target_map[key]
        target_map_reves[value] = key
    if targetColumnName != None: #El dataset contiene el target para comparar cuantas predicciones correctas e incorrectas
        targetColumn = testX[targetColumnName]
        del testX[targetColumnName]
        #print(testX)
        totalInstancias = len(targetColumn)
        aciertos = 0
        try:
            predictions = clf.predict(testX)
            predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
            for index,value in targetColumn.iteritems():
                if value == predictions[index]:
                    aciertos += 1
            print("\n\tEstadisticas de las predicciones")
            print("\t--------------------------------")
            print(f"\t\tAciertos: {aciertos}")
            print(f"\t\tErrores: {totalInstancias-aciertos}")
            print("\n\tInstancia\tValor real\tValor predecido")
            print("\t---------\t----------\t---------------")
            for index,value in targetColumn.iteritems():
                print(f"\t{index}\t\t{target_map_reves[value]} \t{target_map_reves[predictions[index]]}")
            print()
        except:
            print("[!] No has usado el mismo vocabulario de lenguaje natural para entrenar y para predecir")
            sys.exit(1)
    else: #Se predice y se muestran las predicciones
        try:
            predictions = clf.predict(testX)
            predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
            print("\n\tPredicciones")
            print("\t------------")
            for index,value in predictions.iteritems():
                print(f"\tInstancia {index} --> {target_map_reves[index]}")

        except:
            print("[!] No has usado el mismo vocabulario de lenguaje natural para entrenar y para predecir")
            sys.exit(1)
    """try:
        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)
        predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
        results_test = testX.join(predictions, how='left')
        print(predictions)
        results_test.to_csv('predicciones.csv', index=False)
    except:
        print("[!] No has usado el mismo vocabulario de lenguaje natural para entrenar y para predecir")
"""
