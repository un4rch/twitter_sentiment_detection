import pandas as pd
import sys
import os
import numpy as np
import time
import nltk
import emot
import emoji
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords', quiet=True) # Palabras que no aportan valor al texto (in, of, and, ...)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

class Preprocessor:

    def __init__(self) -> None:
        #self.ml_dataset = pMl_dataset
        pass
    
    def __str__(self) -> str:
        print('print for print(Preprocessor object)')

    def dropColumns(self, pMl_dataset, pColNames):
        #Excluir las columnas que no nos interesen
        for colName in pColNames:
            if colName in pMl_dataset:
                pMl_dataset = pMl_dataset.drop(columns=colName)
        return pMl_dataset
    
    def dropNullTargetRows(self, pMl_dataset, pTargetColumn):
        # Eliminar las lineas donde el valor de target sea desconocido (null)
        if pTargetColumn != None:
            pMl_dataset = pMl_dataset[~pMl_dataset[pTargetColumn].isnull()]
        return pMl_dataset

    def _imputeMissingValue(self, pMl_dataset, pColName, pImputeMethod, pConstantValue): # Private method
        tipoDato = pMl_dataset[pColName].dtype
        if pImputeMethod == 'MEAN':
            if tipoDato != 'object':
                imputeValue = pMl_dataset[pColName].mean()
            else:
                print(f"[!] No se puede calcular la media de la columna \"{pColName}\" ya que contiene algun texto")
                sys.exit(1)
        elif pImputeMethod == 'MEDIAN':
            if tipoDato != 'object':
                imputeValue = pMl_dataset[pColName].median()
            else:
                print(f"[!] No se puede calcular la mediana de la columna \"{pColName}\" ya que contiene algun texto")
                sys.exit(1)
        elif pImputeMethod == 'MODE':
            try:
                imputeValue = pMl_dataset[pColName].value_counts().index[0]
            except:
                del pMl_dataset[pColName]
        elif pImputeMethod == 'CONSTANT':
            try:
                imputeValue = float(pConstantValue)
            except:
                print('[!] Alguno de los metodos de imputacion CONSTANT no tiene valor numerico')
                sys.exit(1)
        else:
            print(f'[!] Metodo de imputacion mal especificado para la columna {pColName}')
            sys.exit(1)
        try:
            pMl_dataset[pColName] = pMl_dataset[pColName].fillna(imputeValue)
        except:
            None
        return pMl_dataset
    
    def imputeMissingValues(self, pMl_dataset, pImputeMethod, pConstantValue):
        #Si el metodo de imputacion es un fichero con cada columna con un metodo de imputacion particular
        if os.path.exists(pImputeMethod):
            #nombreCol,method,[value]
            with open(pImputeMethod, 'r') as file:
                for line in file.readlines():
                    line = line.rstrip('\n').split(',')
                    colName = line[0]
                    imputeMethod = line[1].upper()
                    if imputeMethod == 'CONSTANT':
                        try:
                            const = float(line[2])
                            pMl_dataset = self._imputeMissingValue(pMl_dataset, colName, imputeMethod, const)
                        except:
                            print('[!] Alguno de los metodos de imputacion CONSTANT no tiene valor numerico')
                            sys.exit(1)
                    else:
                        pMl_dataset = self._imputeMissingValue(pMl_dataset, colName, imputeMethod, None)
        else:
            imputeMethod = pImputeMethod.upper()
            for colName in pMl_dataset:
                if imputeMethod == 'CONSTANT':
                    pMl_dataset = self._imputeMissingValue(pMl_dataset, colName, imputeMethod, pConstantValue)
                else:
                    pMl_dataset = self._imputeMissingValue(pMl_dataset, colName, imputeMethod, None)
        return pMl_dataset
    
    def parseDataTypes(self, pMl_dataset):
        for colName in pMl_dataset:
            tipoDato = pMl_dataset[colName].dtype
            if tipoDato == np.dtype('M8[ns]') or (hasattr(tipoDato, 'base') and tipoDato.base == np.dtype('M8[ns]')):
                pMl_dataset[colName] =  int(time.mktime(pMl_dataset[colName].timetuple()))
            elif tipoDato == np.dtype("object"):
                pMl_dataset[colName] = pMl_dataset[colName].apply(str) #Convertir a unicode
            elif tipoDato == np.dtype("int64") or tipoDato == np.dtype("float64"):
                pMl_dataset[colName] = pMl_dataset[colName].astype('double') #Convertir a double
        return pMl_dataset
    
    def _rescaleColumn(self, pMl_dataset, pColName, pRescaleOption):
        tipoDato = pMl_dataset[pColName].dtype
        if tipoDato == np.dtype('object'): #Ignorar columnas que sean texto para no escalarlas
            None
        elif pRescaleOption == 'MAX':
            scale = max
            shift = 0
        elif pRescaleOption == 'MINMAX':
            min = float(pMl_dataset[pColName].min())
            max = float(pMl_dataset[pColName].max())
            scale = max - min
            shift = min
        elif pRescaleOption == 'Z-SCALE':
            shift = float(pMl_dataset[pColName].mean())
            scale = float(pMl_dataset[pColName].std())
        else:
            print('[!] Alguno de los metodos de rescalado es incorrecto')
            sys.exit(1)
        if scale == 0:
            del pMl_dataset[pColName] #Eliminar la columna porque no tiene varianza
        else:
            pMl_dataset[pColName] = (pMl_dataset[pColName] - shift).astype(np.float64) / scale
        return pMl_dataset

    def rescaleData(self, pMl_dataset, pRescaleOption):
        if os.path.exists(pRescaleOption):
            with open(pRescaleOption, 'r') as file: #colname,escaleMethod
                for line in file.readlines():
                    line = line.rstrip('\n').split(',')
                    rescaleMethod = line[1].upper() # Convertir el metodo de escalado a mayusculas
                    colName = line[0]
                    pMl_dataset = self._rescaleColumn(pMl_dataset, colName, rescaleMethod)

        else:
            rescaleMethod = pRescaleOption.upper()
            for colName in pMl_dataset:
                pMl_dataset = self._rescaleColumn(pMl_dataset, colName, rescaleMethod)
        return pMl_dataset

    def convertTargetToClassifyInt(self, pMl_dataset, pTargetColumn):
        target_map = {}
        if pTargetColumn == None:
            None
        else:
            targetValues = pMl_dataset[pTargetColumn].dropna().unique()
            for i in range(0, len(targetValues)):
                valor = str(targetValues[i])
                target_map[valor] = i
            pMl_dataset_copy = pMl_dataset.copy()
            pMl_dataset_copy[pTargetColumn] = pMl_dataset[pTargetColumn].map(str).map(target_map)
            pMl_dataset = pMl_dataset_copy
        #del pMl_dataset[targetColumn]
        return pMl_dataset, target_map
    
    def preprocessDataset(self, pMl_dataset, pTargetColumn, pAlgorithm, pExcludedColumns, pImputeOption, pRescaleOption, pNLcolumns, pNLtechnique, pTarinTest):
        # Eliminar columnas que no interesan
        if pExcludedColumns != None:
            columnNames = pExcludedColumns.split(',')
            pMl_dataset = self.dropColumns(pMl_dataset ,columnNames)
    
        # Eliminar las lineas donde el valor del TARGET sea desconocido
        pMl_dataset = self.dropNullTargetRows(pMl_dataset, pTargetColumn)

        # Imputar valores que falten
        if pImputeOption != None:
            if pImputeOption.split(',')[0] == 'CONSTANT':
                pMl_dataset = self.imputeMissingValues(pMl_dataset,pImputeOption.split(',')[0], pImputeOption.split(',')[1])
            else:
                pMl_dataset = self.imputeMissingValues(pMl_dataset,pImputeOption, None)

        # Convertir los datos del dataset a float o unicode
        pMl_dataset = self.parseDataTypes(pMl_dataset)
    
        # Escalar los valores
        if pAlgorithm == 'knn' and pRescaleOption != None:
            pMl_dataset = self.rescaleData(pMl_dataset, pRescaleOption)

        # Enumerar los valores de la columna TARGET para clasificarlos por numeros
        pMl_dataset, target_map = self.convertTargetToClassifyInt(pMl_dataset, pTargetColumn)

        # Preprocesar lenguaje natural
        # escogemos la técnica de preproceso de lenguaje natural
        if pTarinTest == "test":
            vectorizador = pickle.load(open(pNLtechnique+".pkl", "rb"))
            for columnaLN in pNLcolumns:
                pMl_dataset[columnaLN] = Preprocessor.preprocesarLenguajeNatural(pMl_dataset[columnaLN])
                columnaTech = vectorizador.transform(pMl_dataset[columnaLN])
                tech_df = pd.DataFrame(columnaTech.toarray(), columns=vectorizador.get_feature_names_out())
                pMl_dataset = pd.concat([pMl_dataset, tech_df], axis=1)
            for columnaLN in pNLcolumns:
                pMl_dataset = pMl_dataset.drop(columnaLN, axis=1)
        elif pTarinTest == "train":
            if pNLtechnique == "tfidf":
                vectorizador = TfidfVectorizer()
            elif pNLtechnique == "bow":
                vectorizador = CountVectorizer()
            # realizamos el preprocesado
            for columnaLN in pNLcolumns:
                pMl_dataset[columnaLN] = Preprocessor.preprocesarLenguajeNatural(pMl_dataset[columnaLN])
                columnaTech = vectorizador.fit_transform(pMl_dataset[columnaLN])
                tech_df = pd.DataFrame(columnaTech.toarray(), columns=vectorizador.get_feature_names_out())
                pMl_dataset = pd.concat([pMl_dataset, tech_df], axis=1)
            for columnaLN in pNLcolumns:
                pMl_dataset = pMl_dataset.drop(columnaLN, axis=1)
            with open(pNLtechnique + '.pkl', 'wb') as f:
                pickle.dump(vectorizador, f)
        
        return pMl_dataset,target_map
    
    def convertirEmojis(texto):  # convierte un emoji en un conjunto de palabras en inglés que lo representan
        texto = emoji.demojize(texto)
        diccionario_emojis = emot.emo_unicode.EMOTICONS_EMO
        for emoticono, texto_emoji in diccionario_emojis.items():
            texto = texto.replace(emoticono, texto_emoji)
        return texto
    
    def normalizarTexto(texto):  # dado un string que contenga palabras, devuelve un string donde todas las letras sean minúsculas
        return(texto.lower())

    def eliminarSignosPuntuacion(texto):  # dado un string, devuelve el mismo string eliminando todos los caracteres que no sean alfabéticos
        textoNuevo = ""
        for caracter in texto:  # por cada caracter en el texto
            if caracter == "_":  # si es una barra baja, entonces se traduce como espacio
                textoNuevo = textoNuevo + " "
            if caracter.isalpha() or caracter == " ":  # si pertenece al conjunto de letras del alfabeto, se engancha a "textoNuevo"
                textoNuevo = textoNuevo + caracter
        return(textoNuevo)
    
    def eliminarStopWords(texto):  # dado un string, elimina las stopwords de ese string
        texto = word_tokenize(texto, language='english')
        textoNuevo = ""
        for palabra in texto:
            if palabra not in stopwords.words('english'):
                textoNuevo = textoNuevo + " " + palabra
        return(textoNuevo)
    
    def aux_lematizar(palabra):
        tag = nltk.pos_tag([palabra])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lematizar(texto):  # dado un string, lematiza las palabras de ese string
        texto = nltk.word_tokenize(texto)
    
        # Inicializar el lematizador
        lemmatizer = WordNetLemmatizer()
        
        # Lematizar cada palabra y agregarla a una lista
        palabras_lematizadas = []
        for palabra in texto:
            pos = Preprocessor.aux_lematizar(palabra)
            palabra_l = lemmatizer.lemmatize(palabra, pos=pos)
            palabras_lematizadas.append(palabra_l)
        
        # Unir las palabras lematizadas en un solo string y devolverlo
        texto_lematizado = ' '.join(palabras_lematizadas)
        return texto_lematizado
    
    def preprocesarLenguajeNatural(pColumna):  # realiza todo el preproceso de un string en el orden correcto
        listaLineas = []
        for index,linea in pColumna.iteritems():
            linea = Preprocessor.convertirEmojis(linea)
            linea = Preprocessor.eliminarSignosPuntuacion(linea)
            linea = Preprocessor.normalizarTexto(linea)
            linea = Preprocessor.eliminarStopWords(linea)
            linea = Preprocessor.lematizar(linea)
            listaLineas.append(linea)
        columnaProcesada = pd.Series(listaLineas)
        return columnaProcesada

