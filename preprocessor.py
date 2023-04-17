import pandas as pd
import sys
import os
import numpy as np
import time

class Preprocessor:

    def __init__(self) -> None:
        #self.ml_dataset = pMl_dataset
        pass
    
    def __str__(self) -> str:
        print('print for print(Preprocessor object)')

    """def dropColumns(self, pMl_dataset, pColNames) -> pd.DataFrame():
        #Excluir las columnas que no nos interesen
        if pColNames != None:
            colNames = pColNames.split(',')
            for colName in colNames:
                if colName in self.ml_dataset:
                    self.ml_dataset = self.ml_dataset.drop(columns=colName)
        return self.ml_dataset"""
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
            imputeValue = pMl_dataset[pColName].value_counts().index[0]
        elif pImputeMethod == 'CONSTANT':
            try:
                imputeValue = float(pConstantValue)
            except:
                print('[!] Alguno de los metodos de imputacion CONSTANT no tiene valor numerico')
                sys.exit(1)
        else:
            print(f'[!] Metodo de imputacion mal especificado para la columna {pColName}')
            sys.exit(1)
        pMl_dataset[pColName] = pMl_dataset[pColName].fillna(imputeValue)
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
    
    def preprocessDataset(self, pMl_dataset, pTargetColumn, pAlgorithm, pExcludedColumns, pImputeOption, pRescaleOption):
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
        
        return pMl_dataset,target_map
    
    def normalizarTexto(texto):
        return(texto.lower())

    def eliminarSignosPuntuacion(texto):
        pass

    def eliminarStopWords(texto):
        pass

    def lematizar(texto):
        pass