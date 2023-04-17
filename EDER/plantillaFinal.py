# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk    #Biblioteca de machine learning de código abierto para Python
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB


#### GENERAR O UTILIZAR MODELO ####
respuestaGU = input("¿Quieres generar un modelo o utilizar uno existente (G/U) ")
#### GENERAR O UTILIZAR MODELO ####

#### ALTORITMO QUE SE VA A UTILIZAR ####
if(respuestaGU.upper() == "G"):
    algoritmo = input("Introduce el ALGORITMO que deseas utilizar entre KNN, Decision Trees y Naive Bayes (KNN / DT / NB): ")
elif(respuestaGU.upper() == "U"):
    algoritmo = input("Introduce el ALGORITMO que deseas has utilizado para generar el modelo (KNN / DT / NB): ")
#### ALTORITMO QUE SE VA A UTILIZAR ####

def llenar_lista(lista, mensaje):
    print("Introduce uno/varios INTEGER en la lista para " + mensaje + " (espacio en blanco para terminar): ")
    while True:
        entrada = input(str(lista) + " <- ")
        if entrada == "":
            break
        lista.append(int(entrada))

    return lista

def llenar_lista_floats(lista, mensaje):
    print("Introduce uno/varios FLOAT en la lista para " + mensaje + " (espacio en blanco para terminar): ")
    while True:
        entrada = input(str(lista) + " <- ")
        if entrada == "":
            break
        lista.append(float(entrada))

    return lista


if(respuestaGU.upper() == "G"):
    ## VARIABLES KNN ##
    if(algoritmo.upper() == "KNN"):
        ks = []
        llenar_lista(ks, "k (número de vecinos) ")
        #ks = [1, 3, 5]
        k=min(ks)     #k = numero MIN de vecinos
        K=max(ks)     #K = numero MAX de vecinos


        ds = []
        llenar_lista(ds, "d (distancia entre vecinos) ")
        #ds = [1, 2]
        d=min(ds)     #d = distancia MIN entre vecinos
        D=max(ds)     #D = distancia MAX

        weights = []
        print("Valores posibles de weights: 'uniform' 'distance' ")
        respuesta = input("\nIntroduce: \n1 para usar 'uniform' \n2 para usar 'distance' \n3 para usar 'uniform' y 'distance'): ")
        if(respuesta == '1'):
            weights = ['uniform']       # Uniform = todos los vecinos más cercanos se ponderan igualmente
        elif(respuesta == '2'):
            weights = ['distance']      # Distance = Los vecinos más cercanos tienen peso mayor que los más lejanos
        elif(respuesta == '3'):
            weights = ['uniform', 'distance']
        print(weights)

        #weights = ['uniform', 'distance']

    ## VARIABLES DECISION TREES ##
    elif(algoritmo.upper() == "DT"):
        maxDepths = []  #md = Num max de niveles permitidos en el árbol (si no se especifica se expande hasta que todas las hojas tengan menos de msl)
        llenar_lista(maxDepths, "maxDepths ")
        #maxDepths = [3, 6, 9]

        minSamplesSplits = []   #mss = Num min de muestras para dividir un nodo interno en 2 hijos. (Entero: num mínimo de muestras. Float: Fracción del num total de muestras.)
        print("\nmin_samples_split must be an integer greater than 1 or a float in (0.0, 1.0] ")
        llenar_lista_floats(minSamplesSplits, "minSamplesSplits ¡¡ entre 0.0 y 1.0 !!")
        llenar_lista(minSamplesSplits, "minSamplesSplits ¡¡ mayor o igual que 2 !!")
        #minSamplesSplits = [1.0, 2]

        minSamplesLeafs = []    #Num min de muestras que se requieren para SER una HOJA. (Entero: num mínimo de muestras. Float: Fracción del num total de muestras.)
        llenar_lista(minSamplesLeafs, "minSamplesLeafs ")
        #minSamplesLeafs = [1, 2]

        numMaxAtrDataiku = input("Número Máximo de Atributos (espacio en blanco para asignarle None):")
        if numMaxAtrDataiku == "":
            numMaxAtrDataiku = None     #Num max de características a considerar al buscar la mejor división. (None = se utilizarán TODAS las features)
        
    ## VARIABLES DECISION TREES ##
    elif(algoritmo.upper() == "NB"):
        esMixedNB = False


p='./'
f="trainHalfHalf.csv"
oFile="output.out"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])  #Imprime la lista de los arg que se han enviado al script en la command line
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:K:d:D:p:f:h',['output=','k=', 'K=', 'd=', 'D=','path=','iFile','h'])
        #getopt = Opciones que se esperan en la command line
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):        #o = Output file
            oFile = arg
        elif opt == '-k':                   #k = Num MIN de vecinos
            k = arg
        elif opt == '-K':                   #k = Num MAX de vecinos
            K = arg
        elif opt ==  '-d':                  #d = Distancia MIN
            d = arg
        elif opt ==  '-D':                  #d = Distancia MAX
            D = arg
        elif opt in ('-p', '--path'):       #p = Input file path
            p = arg
        elif opt in ('-f', '--file'):       #f = input file name
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItemsMIN \n -K numberOfItemsMAX \n -d distanceParameterMIN \n -D distanceParameterMAX \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':       #Si está en la carpeta actual:
        iFile=p+str(f)
    else:               #Si no:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):   #Asegura que los objetos se conviertan en unicode
        if sys.version_info < (3, 0):   #Si la v de Python es < 3.0
            if isinstance(x, str):      #Si es un str
                return unicode(x, 'utf-8')  #Se convierte a unicode usando UTF-8
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como 
    #la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado


    respuesta = input("¿Quieres mantener o eliminar alguna columna? (SI/NO) ")

    if(respuesta.upper() == "SI"):
        respuesta2 = input("Introduce M para mantener o E para eliminar (M/E) ")
        if respuesta2.upper() == "M":
            columnas_a_preservar = input("Introduce los nombres de las columnas que deseas preservar, separados por comas: ").split(",")
            ml_dataset = ml_dataset[columnas_a_preservar]
        elif respuesta2.upper() == "E":
            columnas_a_eliminar = input("Introduce los nombres de las columnas que deseas eliminar, separados por comas: ").split(",")
            ml_dataset = ml_dataset.drop(columnas_a_eliminar, axis=1)

    print(ml_dataset)


    if(respuestaGU == "G"):

        TARGET = input("Introduce el nombre de la columna objetivo: ")
        
        #Todas las col categoriales:
        if ml_dataset[TARGET].dtype == 'object':    #si el TARGET es de tipo object (categorial o texto)
            categorical_features = ml_dataset.select_dtypes(include=['object']).drop(TARGET, axis=1).columns.tolist()
        else:
            categorical_features = ml_dataset.select_dtypes(include=['object']).columns.tolist()
        print("Checkpoint 1, Categoriales:", categorical_features)
    
        #Todas las col numericas:
        if ml_dataset[TARGET].dtype in ['float64', 'int64']:    #si el TARGET es de tipo numerico
            numerical_features = ml_dataset.select_dtypes(include=['float64', 'int64']).drop(TARGET, axis=1).columns.tolist()   #se excluye
        else:
            numerical_features = ml_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()    #no se excluye
        print("Checkpoint 2, Numéricas:", numerical_features)


    

        #E:
        text_features = []  #TODO
        for feature in categorical_features:    #CATEGORIALES
            # Convierte los valores a unicode, se guarda de vuelta en la misma columna, util cuando hay caracteres especiales o acentos
            ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

        for feature in text_features:   #TEXTO
            ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

        for feature in numerical_features:  #NUMERICOS
            if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                    hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
                    #Si el tipo de datos es (datetime64[ns], que es igual a M8[ns]), es decir, si contiene fechas
                    #Si el atr base es M8 (base se usa para datos compuestos con varios subtipos)
                #Convierte las fechas en valores de tiempo UNIX, sobreescribiendo la propia columna
                ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
            else:
                #Se convierte la columna en double
                ml_dataset[feature] = ml_dataset[feature].astype('double')

        print("CHECKPOINT 3", ml_dataset)
        #num_classes = ml_dataset[TARGET].nunique()  #Esta variable se utilizará en KNN para hacer las métricas según si es BINARIA o MULTICLASE
        num_classes = len(ml_dataset[TARGET].unique())
        print("Se han detectado", num_classes, "clases")

        #### BINARIO #### (meterlos a mano si no funciona)
        if num_classes == 2:
            #target_map = {'0': 0, '1': 1}      #Mapear los valores de los objetivos originales a nuevos valores
            target_values = ml_dataset[TARGET].unique() # obtén los valores únicos de la variable objetivo
            target_map = {str(val): i for i, val in enumerate(target_values)} # crea un diccionario que mapea los valores a números enteros
            ## ¡CUIDADO! Puede no ser un str ##
            ml_dataset['__target__'] = ml_dataset[TARGET].map(str).map(target_map)

        #### MULTICLASE ####
        else:
            target_map = {}
            unique_targets = sorted(ml_dataset[TARGET].unique())
            for i, target_class in enumerate(unique_targets):
                target_map[target_class] = i
            ml_dataset['__target__'] = ml_dataset[TARGET].map(target_map)    #Aplicar el mapeo target_map a la columna TARGET


        del ml_dataset[TARGET]    #Elimina la col original TARGET

        # Remove rows for which the target is unknown (NULL).
        ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
        print(f)    #Nom del archivo
        print("CHECKPOINT 4", ml_dataset)
        print("Primeros 5 valores del DATASET: \n", ml_dataset.head(5))   #Los primeros 5 registros del cjto de datos


        train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
        #80% TRAIN  ,   20% TEST    ,   RANDOM_STATE = semilla aleatoria    ,   STRATIFY = estratificación basada en los valores objetivo
        print("Primeros 5 valores de TRAIN: \n", train.head(5))
        print(train['__target__'].value_counts())   #Nº veces que cada valor objetivo aparece en ENTRENAMIENTO
        print(test['__target__'].value_counts())    #Nº veces que cada valor objetivo aparece en TEST
        # Estos print sirven para verificar la estratificacion
        # y para ver si los objetivos están distribuidos DE MANERA SIMILAR en train y test

        drop_rows_when_missing = []


        #Todas las col que puedan tener NA
        METODO_IMPUTACION = input("Introduce el método de IMPUTACIÓN que deseas utilizar (MEAN / MEDIAN / CREATE_CATEGORY / MODE / CONSTANT): ")
        impute_when_missing = [{'feature': col, 'impute_with': METODO_IMPUTACION} for col in ml_dataset.columns if col != "__target__"] #excluimos a __target__




        # E: Eliminar filas de TRAIN y TEST que tienen valores faltanes N/A en las caracteristicas almacenadas en DROP_ROWS_WHEN_MISSING
        for feature in drop_rows_when_missing:
            train = train[train[feature].notnull()]     #Selecciona las NO NULAS de train, y se sobreescribe el cjto con las filas seleccionadas
            test = test[test[feature].notnull()]        #Selecciona las NO NULAS de test, y lo mismo
            print('Dropped missing records in %s' % feature)    #Se han eliminado los registros faltantes para la caracteristica actual

        # E: Imputar valores faltantes en TRAIN y TEST en las fetures almacenadas en IMPUTE_WHEN_MISSING
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':        #cada feature tiene un metodo asociado
                v = train[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = train[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY'                                     #una categoria nueva (en este caso NULL_CATEGORY)
            elif feature['impute_with'] == 'MODE':
                v = train[feature['feature']].value_counts().index[0]   #la moda
            elif feature['impute_with'] == 'CONSTANT':
                v = feature['value']                                    #un valor kte
            train[feature['feature']] = train[feature['feature']].fillna(v)     #Reemplaza los N/A por el valor v
            test[feature['feature']] = test[feature['feature']].fillna(v)       #Reemplaza los N/A por v

            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))
            #Se han imputado valores faltantes para la feature actual, junto con el valor de imputacion utilizado



        if(algoritmo.upper() == "KNN"):     #Decision Trees no necesita escalar
            METODO_ESCALADO = input("Introduce el método de ESCALADO que deseas utilizar (AVGSTD / MINMAX): ")
            rescale_features = {col: METODO_ESCALADO for col in ml_dataset.columns if col != "__target__"}  #excluir a __target__ del escalado

            # E: Reescalar ciertas features el TRAIN y TEST. Se realiza para asegurar que todas las features tengan la misma escala y distr
            for (feature_name, rescale_method) in rescale_features.items():
                #par clave-valor, la clave es el nombre y el valor es el metodo de reescalado a utilizar (en este caso MINMAX o ZSCORE)
                if rescale_method == 'MINMAX':
                    _min = train[feature_name].min()
                    _max = train[feature_name].max()
                    scale = _max - _min
                    shift = _min
                else:   # metodo = ZSCORE (AVGSTD)
                    shift = train[feature_name].mean()
                    scale = train[feature_name].std()
                if scale == 0.:     #significa que la feature no tiene varianza
                    del train[feature_name]
                    del test[feature_name]
                    print('Feature %s was dropped because it has no variance' % feature_name)
                else:               #tiene varianza
                    print('Rescaled %s' % feature_name)
                    train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale      #reescalado
                    test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale        #reescalado






        trainX = train.drop('__target__', axis=1)   #contienen todas las col excepto TARGET
        #trainY = train['__target__']

        testX = test.drop('__target__', axis=1)
        #testY = test['__target__']

        trainY = np.array(train['__target__'])      #contienen TARGET
        testY = np.array(test['__target__'])

        # E: Aplicar una tecnica de submuestreo (undersampling) para equilibrar la proporcion de las clases en la
        #var objetivo (TARGET), la cual puede estar desbalanceada
        sampling_strategy = {}
        if(num_classes == 2):
            sampling_strategy = 0.5     #la mayoria va a estar representada el doble de veces
        else:
            for i in range(num_classes):
                sampling_strategy[i] = 1    #introducir otro si es necesario

        undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)    #TODO   Otra opción: sampling_strategy='majority'
        #oversample = RandomOverSampler(sampling_strategy=sampling_strategy)    #TODO  Otra opción: sampling_strategy='minority'

        # estrategia de submuestreo = 0'5 (se va a reducir la cant de registros en la clase mayoritaria para que 
        #quede representada el doble de veces que la clase minoritaria)
        #RandomUnderSampler toma una muestra aleatoria de la clase mayoritaria para equilibrarla con la cant de registros de la minoritaria


        #Se almacenan los NUEVOS cjtos de TRAIN y TEST en las variables:
        trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
        testXUnder,testYUnder = undersample.fit_resample(testX, testY)







        def entrenarYMostrarDatos():
            global best_clf
            global clf, trainX, trainY, testX, target_map, TARGET, testY, num_classes, algoritmo, results
            global k, d, w, max_f1, best_k, best_d, best_w  #knn
            global maxDepths, minSamplesSplits, minSamplesLeafs, numMaxAtrDataiku   #dt
            global best_md, best_mss, best_msl  #dt
            global esMixedNB    #nb
            # E: Establecer el param class_weight. Se utiliza para manejar el desbalance de clases en el cjto de datos.
            # Balanced: Se ajustan los pesos de las clases inversamente proporcional a su frec en el cjto de TRAIN.
            # de esta forma, las clases minoritarias tienen mas peso en la funcion de perdida de clasificador, lo que les da mas importancia en la predicción final.
            # Util para cjtos de datos con clases desbalanceadas.
            clf.class_weight = "balanced"


    # E: Entrenar el modelo clasificador con TRAIN.
    # Fit = ajustar el modelo a los datos de TRAIN, es decir, se encuentra el patron en los datos TRAIN y se establecen los param del modelo para hacer predicciones
            clf.fit(trainX, trainY)






# Build up our result dataset -> En este paso, se realiza la predicción del modelo entrenado en el cjto de TEST

# The model is now trained, we can apply it to our test set:

            predictions = clf.predict(testX)        #Predecir var objetivo a partir de las var predictoras (testX)
            probas = clf.predict_proba(testX)       #Obtener las probabilidades estimadas de cada clase de la var objetivo para cada observación del cjto TEST

            predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')    #Crear serie de Pandas con las predicciones
            cols = [
                u'probability_of_value_%s' % label      #cada col contiene ese str seguido del valor objetivo 
                for (_, label) in sorted([(int(target_map[label]), label) for label in target_map]) #itera a través de target y los ordena según su valor numérico
            ]
            probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols) #se crea un DF de Pandas con las prob


# Build scored dataset
            results_test = testX.join(predictions, how='left')      #join para unir las col de predictions y probabilities al dataset testX
            results_test = results_test.join(probabilities, how='left')
            results_test = results_test.join(test['__target__'], how='left')    #Se agrega la col de la var objetivo original
            results_test = results_test.rename(columns= {'__target__': TARGET})   #Se renombra a TARGET

    #Se imprime una pequeña muestra (de 6) de las predicciones realizadas en TEST. 
    #Se itera sobre las etiquetas reales y las predichas y se imprimen por pantalla las primeras 6 parejas (real, predicha)
            #i=0
            #for real,pred in zip(testY,predictions):
            #    print(real,pred)
            #    i+=1
            #    if i>5:
            #        break

            #Para cada clase del modelo entrenado:
            print(f1_score(testY, predictions, average=None))
            print(classification_report(testY,predictions))     #Métricas de precision, exhaustividad y F1 para cada clase, asi como el promedio ponderado de todas
            print(confusion_matrix(testY, predictions, labels=[1,0])) #Matriz de confusión

            #Tarea: Archivo CSV y MEJOR RESULTADO
            if(algoritmo.upper() == "NB"):
                if(esMixedNB):
                    metodoNB = 'Mixed-NB'
                else:
                    metodoNB = 'Gaussian-NB'
            
            if num_classes == 2:    #Si la clasificación es BINARIA, num_classes se ha declarado antes de eliminar la columna TARGET
                # Calcular las métricas de evaluación
                acc = accuracy_score(testY, predictions)
                precision = precision_score(testY, predictions)
                recall = recall_score(testY, predictions)
                f1 = f1_score(testY, predictions)
                print("Acc ", acc)
                print("Prec", precision)
                print("Rec", recall)
                print("F1", f1)

                if(algoritmo.upper() == "KNN"):
                    # Almacenar los resultados en una lista de diccionarios
                    results.append({'n_neighbors(k)': k, 'distance(d)': d, 'weights': w,    #append permite agregar un elemento al final de la lista
                                    'accuracy': acc, 'precision': precision, 
                                    'recall': recall, 'f1_score': f1})

                    # Comprobar si tiene el f1 más alto
                    if(f1 > max_f1):
                        best_clf = clf
                        max_f1, best_k, best_d, best_w = f1, k, d, w

                elif(algoritmo.upper() == "DT"):
                    results.append({'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,    #append permite agregar un elemento al final de la lista
                                    'numMaxAtrDataiku': numMaxAtrDataiku, 
                                    'accuracy': acc, 'precision': precision, 
                                    'recall': recall, 'f1_score': f1})
                    
                    if(f1 > max_f1):
                        best_clf = clf
                        max_f1, best_md, best_mss, best_msl = f1, max_depth, min_samples_split, min_samples_leaf
                
                elif(algoritmo.upper() == "NB"):
                    results.append({'accuracy': acc, 'precision': precision, 
                                    'recall': recall, 'f1_score': f1, 
                                    'Naive Bayes': metodoNB})
                    
                    if(f1 > max_f1):
                        best_clf = clf
                        max_f1 = f1


            else:   #Si la clasificación es MULTICLASE
                macro = f1_score(testY, predictions, average='macro')
                micro = f1_score(testY, predictions, average='micro')
                weighted = f1_score(testY, predictions, average='weighted')
                accuracy = accuracy_score(testY, predictions)

                if(algoritmo.upper() == "KNN"):
                    # Almacenar los resultados en una lista de diccionarios
                    results.append({'n_neighbors(k)': k, 'distance(d)': d, 'weights': w, 
                                    'macroF1': macro, 
                                    'microF1': micro, 'weightedF1': weighted,
                                    'accuracy': accuracy})

                    # Comprobar si tiene el macro/micro/weighted más alto
                    if(macro > max_f1):     ## Cambiar la variable por la que queramos tener en cuenta. # MACRO: Eval equitativa clases sin tener en cuenta cant
                        best_clf = clf
                        max_f1, best_k, best_d, best_w = macro, k, d, w                                    # MICRO: Eval global + teniendo en cuenta todas las clases y cant
                                                                                                                # WEIGHTED: Eval equitativa clases teniendo en cuenta cant
                elif(algoritmo.upper() == "DT"):
                    results.append({'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,    #append permite agregar un elemento al final de la lista
                                    'numMaxAtrDataiku': numMaxAtrDataiku, 
                                    'macro': macro, 
                                    'micro': micro, 'weighted': weighted})
                    
                    if(macro > max_f1):
                        best_clf = clf
                        max_f1, best_md, best_mss, best_msl = macro, max_depth, min_samples_split, min_samples_leaf

                elif(algoritmo.upper() == "NB"):
                    results.append({'macro': macro, 
                                    'micro': micro, 'weighted': weighted, 
                                    'Naive Bayes': metodoNB})
                    
                    if(macro > max_f1):
                        best_clf = clf
                        max_f1 = macro


        ## ESTE LLAMA A LA FUNCIÓN DE ARRIBA ##
        best_clf = None
        results = []    #Aquí se almacenan los resulados de cada combinación de parámetros KNN o DT
        max_f1 = 0
        if(algoritmo.upper() == "KNN"):
            # E: Crear un clasificador KNN (K-Nearest Neighbors) con los siguientes param:
            # Tarea: for dentro de otro for dentro de otro for
            best_k, best_d, best_w = 0, 0, 0

            for k in ks:
                for d in ds:
                    for w in weights:
                        pesos = 'uniform'
                        print("k:", k, " d:", d, " w:", w)
                        clf = KNeighborsClassifier(n_neighbors=k,       #Nº de vecinos que se usarán para clasificar una instancia
                                                weights=w,        #Tipo de peso que se asignará a cada vecino (Uniform = todos el mismo peso)
                                                algorithm='auto',         #Algoritmo utilizado para buscar los vecinos mas cercanos (Auto = algot mas apropiado segun los datos de TRAIN)
                                                leaf_size=30,             #Tamaño de las hojas en el arbol KD-Tree. Se utiliza para buscar los vecinos mas cercanos
                                                p=d)                      #Distancia que se usará para buscar los vecinos mas cercanos. p=2 -> Distancia euclidiana
                        entrenarYMostrarDatos()
        elif(algoritmo.upper() == "DT"):
            # E: Realizar los mismos pasos pero esta vez seleccionando como algoritmo los Arboles de Decisión
            best_md, best_mss, best_msl = 0, 0, 0

            for max_depth in maxDepths:
                for min_samples_split in minSamplesSplits:
                    for min_samples_leaf in minSamplesLeafs:
                        print("md:", max_depth, " mss:", min_samples_split, " msl:", min_samples_leaf)
                        clf = DecisionTreeClassifier(max_depth=max_depth, 
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf, 
                                        max_features=numMaxAtrDataiku)
                        entrenarYMostrarDatos()

        elif(algoritmo.upper() == "NB"):
            #clf = GaussianNB()     # Diria que esto no hace falta
            #entrenarYMostrarDatos()

            ## MIXED NAIVE BAYES ##
            esMixedNB = True
            clf = MixedNB()
            entrenarYMostrarDatos()

            ## GAUSSIAN NAIVE BAYES ##
            esMixedNB = False
            discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  #Se utiliza comunmente con GaussianNB, ya que GaussianNB asume distribución normal, por lo que la discretización de las continuas puede mejorar su desempeño
            clf = GaussianNB()

            trainX_ant = trainX   #por si acaso
            trainX = discretizer.fit_transform(trainX)
            #testX_ant = testX  #TODO solucionar el error de transform
            #testX = discretizer.transform(testX)

            entrenarYMostrarDatos()

            #Comprobar si el mejor clf es GaussianNB o MixedNB
            if(clf != best_clf):    #Si el clf (el ultimo es gaussian) es distinto al mejor clf
                esMixedNB = True
            else:
                esMixedNB = False




                                                                                                        
        #T: Convertirlo al CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('results.csv', index=False)

        #Tarea: Repetir el experimento para el mejor modelo y salvarlo
        if(algoritmo.upper() == "KNN"):
            print("\n\n\n\nSe va a salvar el modelo con los siguientes parámetros: k=",best_k, ", d=", best_d, ", w=", best_w)
        elif(algoritmo.upper() == "DT"):
            print("\n\n\n\nSe va a salvar el modelo con los siguientes parámetros: max_depth=",best_md, ", min_samples_split=", best_mss, ", min_samples_leaf=", best_msl, ", numMaxAtrDataiku=", numMaxAtrDataiku)
        elif(algoritmo.upper() == "NB"):
            if not esMixedNB:
                print("\n\n\n\nSe va a salvar el siguiente modelo: Naive Bayes -> Gaussian Naive Bayes")
                with open("discretizer.pkl", "wb") as f:    #Guardamos el discretizador porque tiene los datos de TRAIN
                    pickle.dump(discretizer, f)
            else:
                print("\n\n\n\nSe va a salvar el siguiente modelo: Naive Bayes -> Mixed Naive Bayes")

        nombreModel = "MejorModelo.sav"
        saved_model = pickle.dump(best_clf, open(nombreModel,'wb'))
        print("Modelo guardado correctamente empleando Pickle")




    elif(respuestaGU.upper() == "U"):     #Si la respuesta es: USAR un modelo existente
        X_nuevo = pd.read_csv(iFile)

        categorical_features = X_nuevo.select_dtypes(include=['object']).columns.tolist()
        print("Checkpoint 1, Categoriales:", categorical_features)
    
        numerical_features = X_nuevo.select_dtypes(include=['float64', 'int64']).columns.tolist()    #no se excluye
        print("Checkpoint 2, Numéricas:", numerical_features)

        text_features = []  #TODO
        for feature in categorical_features:
            X_nuevo[feature] = X_nuevo[feature].apply(coerce_to_unicode)
        for feature in text_features:
            X_nuevo[feature] = X_nuevo[feature].apply(coerce_to_unicode)
        for feature in numerical_features:
            if X_nuevo[feature].dtype == np.dtype('M8[ns]') or (
                    hasattr(X_nuevo[feature].dtype, 'base') and X_nuevo[feature].dtype.base == np.dtype('M8[ns]')):
                X_nuevo[feature] = datetime_to_epoch(X_nuevo[feature])
            else:
                X_nuevo[feature] = X_nuevo[feature].astype('double')



        METODO_IMPUTACION = input("Introduce el método de IMPUTACIÓN que deseas utilizar (MEAN / MEDIAN / CREATE_CATEGORY / MODE / CONSTANT): ")
        impute_when_missing = [{'feature': col, 'impute_with': METODO_IMPUTACION} for col in X_nuevo.columns]
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v = X_nuevo[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = X_nuevo[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY' 
            elif feature['impute_with'] == 'MODE':
                v = X_nuevo[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v = feature['value'] 
            X_nuevo[feature['feature']] = X_nuevo[feature['feature']].fillna(v)

            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

        
        if(algoritmo.upper() == "KNN"):
            METODO_ESCALADO = input("Introduce el método de ESCALADO que deseas utilizar (AVGSTD / MINMAX): ")
            rescale_features = {col: METODO_ESCALADO for col in X_nuevo.columns}

            for (feature_name, rescale_method) in rescale_features.items():
                if rescale_method == 'MINMAX':
                    _min = X_nuevo[feature_name].min()
                    _max = X_nuevo[feature_name].max()
                    scale = _max - _min
                    shift = _min
                else:   # metodo = ZSCORE (AVGSTD)
                    shift = X_nuevo[feature_name].mean()
                    scale = X_nuevo[feature_name].std()
                if scale == 0.:
                    del X_nuevo[feature_name]
                    print('Feature %s was dropped because it has no variance' % feature_name)
                else:
                    print('Rescaled %s' % feature_name)
                    X_nuevo[feature_name] = (X_nuevo[feature_name] - shift).astype(np.float64) / scale



    
        nombreModel = input("Introduce el nombre del modelo sin el .sav (tiene que estar en la misma ruta): ") + ".sav"
        clf = pickle.load(open(nombreModel, 'rb'))

        if(algoritmo.upper() == "KNN"):
            print("K:", clf.n_neighbors, " D:", clf.p, " W:", clf.weights)
        elif(algoritmo.upper() == "DT"):
            print("max_depth:", clf.max_depth, " min_samples_split:", clf.min_samples_split, " min_samples_leaf:", clf.min_samples_leaf, " max_features:", clf.max_features)
        elif(algoritmo.upper() == "NB"):
            #TODO comprobar si clf es nb normal o nb mixed, quizas con una variable global -> creo que no se puede
            respuestaNB = input("¿El modelo es Gaussian Naive Bayes? (SI / NO) ")
            if(respuestaNB.upper() == "SI"):   #Gaussian Naive Bayes
                #discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
                with open("discretizer.pkl", "rb") as f:
                    discretizer = pickle.load(f)
                X_nuevo = discretizer.transform(X_nuevo)  #Es importante discretizarlos (de la misma forma del modelo) para que los resultados sean precisos
        
        resultado = clf.predict(X_nuevo)

        print(resultado)

        resultado_df = pd.DataFrame(resultado)
        resultado_df.to_csv('prediccion.csv', index=False)



print("bukatu da")
