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
from gensim import corpora

iFile = None
inputModelName = None
targetColumn = None
targetColumnName = None
algorithms = ['knn', 'decision tree', 'naive bayes', 'logistic regression', 'random forest']
algorithm = None
rescaleOption = None
imputeOption = 'mode'
excludedColumns = None
NLcolumns = ["text"]
NLtechnique = None
NLvectorizer = None
switch = None
airline = None
sentiment = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'t:m:f:ha:r:i:e:v:x:n:c:',['target=','model=','testFile=','h', 'algorithm=', 'rescale=', 'impute=', 'exclude=', 'vectorize=', 'switch=', 'natural-language=', 'clustering'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)

    for opt,arg in options:
        if opt in ('-f', '--file'):
            iFile = arg
        elif opt in ('-m', '--model'):
            inputModelName = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)
        elif opt in ('-t', '--target'):
            targetColumnName = arg
        elif opt in ('-a', '--algorithm'):
            algorithm = int(arg)
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
        elif opt in ('-x', '--switch'):
            switch = arg
            if switch == "on":
                switch = True
            elif switch == "off":
                switch = False
            else:
                print('[!] Switch must be "on" (translate emojis to natural language) or "off" (delete emojis)')
                sys.exit(0)
        elif opt in ('-n', '--natural-language'):
            NLcolumns = arg.split(",")
        elif opt in ('-c', '--clustering'):
            arg = str(arg).upper()
            if NLcolumns == None:
                print("[!] Debes especificar las columnas que tienen texto con el parametro -n")
                sys.exit(1)
            if arg == "T":
                clustering = True
            elif arg == "F":
                clustering == False
            

    if iFile == None or inputModelName == None:
        print('[!] Introduce input file o modelo')
        sys.exit(1)
    if NLcolumns != None and NLtechnique == None:
        print('[!] Introduce la tecnica de vectorizado tfidf o bow')
        sys.exit(1)
    if clustering:
        shift = 2
        for i,value in enumerate(sys.argv[1:]):
            if value == '-c' or value == '--clustering':
                values = sys.argv[1:][i+shift:i+shift+2]
                airline = values[0]
                sentiment = values[1]
        if airline == None or sentiment == None:
            print("[!] Debes especificar una aerolinea y un sentimiento")
            sys.exit(1)
    
    clf = pickle.load(open(inputModelName+"_clf.pkl", 'rb'))
    lda_model = pickle.load(open(inputModelName+"_lda.pkl", 'rb'))

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    print("[*] Cargando el dataset...")
    # Cargar el fichero de datos en un dataset de pandas
    testX = pd.read_csv(iFile)
    print("[*] Preprocesando dataframe...")
    preprocessor = Preprocessor()
    if targetColumnName != None: #El dataset contiene el target para comparar cuantas predicciones correctas e incorrectas
        ml_dataset_classification = testX[[targetColumnName]+NLcolumns]
        ml_dataset_classification = preprocessor.preprocessDataset(ml_dataset_classification, targetColumnName, algorithms[algorithm], excludedColumns, imputeOption, rescaleOption, NLcolumns, NLtechnique, "test", switch)
        ml_dataset_classification, target_map = preprocessor.convertTargetToClassifyInt(ml_dataset_classification, targetColumnName)
        targetColumn = ml_dataset_classification[targetColumnName]
        ml_dataset_classification = ml_dataset_classification.drop([targetColumnName], axis=1)
        target_map_reves = {}
        for key in target_map:
            value = target_map[key]
            target_map_reves[value] = key
        #del testX[targetColumnName]
        #print(testX)
        print("[*] Haciendo predicciones...")
        totalInstancias = len(targetColumn)
        aciertos = 0
        try:
            predictions = clf.predict(ml_dataset_classification)
            predictions = pd.Series(data=predictions, index=ml_dataset_classification.index, name='predicted_sentiment')
            ml_dataset_classification['predicted_sentiment'] = predictions
            predictions_sentiments = []
            for value in predictions:
                predictions_sentiments.append(target_map_reves[value])
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
        ml_dataset_classification = testX[NLcolumns]
        ml_dataset_classification = preprocessor.preprocessDataset(ml_dataset_classification, targetColumnName, algorithms[algorithm], excludedColumns, imputeOption, rescaleOption, NLcolumns, NLtechnique, "test", switch)
        print("[*] Haciendo predicciones...")
        try:
            predictions = clf.predict(ml_dataset_classification)
            predictions = pd.Series(data=predictions, index=ml_dataset_classification.index, name='predicted_sentiment')
            ml_dataset_classification['predicted_sentiment'] = predictions
            print("\n\tPredicciones")
            print("\t------------")
            for index,value in predictions.iteritems():
                print(f"\tInstancia {index} --> {value}")
            print(testX)

        except:
            print("[!] No has usado el mismo vocabulario de lenguaje natural para entrenar y para predecir")
            sys.exit(1)
    predictedTargetColumnName = 'predicted_sentiment'
    testX[predictedTargetColumnName] = predictions_sentiments
    ml_dataset_clustering = preprocessor.preprocessEvolved(testX, targetColumnName, excludedColumns, imputeOption, NLcolumns, NLtechnique, "train", switch, airline, sentiment)
    docs = ml_dataset_clustering[NLcolumns].values.tolist()
    texts = [[word for word in document[0].lower().split()] for document in docs]
    diccionario= corpora.Dictionary(texts)
    corpus = [diccionario.doc2bow(text) for text in texts]
    predicciones = []
    for doc in corpus:
        topic_probs = lda_model.get_document_topics(doc)
        predicciones.append(max(topic_probs, key=lambda x: x[1])[0])
    ml_dataset_clustering['predicted_topic'] = predicciones
    ml_dataset_clustering = ml_dataset_clustering[ml_dataset_clustering[predictedTargetColumnName] == sentiment]
    ml_dataset_clustering.to_csv(airline+"_"+sentiment+".csv", index=False)
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
