#region imports
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import os 
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import List
#endregion

def error_log(linea: int, e: Exception, funcion: str)->None:
    """Función para imprimir los errores de ejecución"""
    print(f'Ha ocurrido algo inesperado en la función {funcion}, revise el reporte de errores.')
    logging.critical(f'Ha ocurrido un eror en la función descripcion, línea {linea}: \n{e}')  

def descripcion(df: pd.DataFrame)->None:
    """Análisis numérico del dataset y sus instancias"""
    try:
        print('Dimensiones del dataset:')
        print(df.shape)
        print(f'\nNombres de las columnas:{df.columns}')
        print('\nTipos de las columnas:')
        print(df.dtypes)
        print('\nRecuento de características nulas')
        print(df.isnull().sum())
        print('\nDescripción del dataset')
        df_desc=df.describe()
        for name,values in df_desc.items():
            print(values)
        print('\nInfo del dataset')
        print(df.info())
        input('')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error_log(exc_tb.tb_lineno,e,'descrpcion')
    
def subplots(df: pd.DataFrame,columns:List[str]):
    try:
        scaler=MinMaxScaler(feature_range=(0,10))
        transformed=scaler.fit_transform(df[columns].values)

        figure, axes = plt.subplots(3, 3,figsize=(10, 10))
        figure.suptitle('Histogramas de las características')
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        i,j=0,0
        k=0
        for col in columns:            
            sns.distplot(transformed[:,k],ax=axes[i,j],axlabel=col)
            if j==2:
                j=0
                i+=1
            else:
                j+=1 
            k+=1
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error_log(exc_tb.tb_lineno,e,'subplot')

def graficar(df: pd.DataFrame)->None:
    """Visualización de las características"""
    try:    
        #region histogramas
        columns=df.columns.drop('Type')
        columns1=columns[:9]
        columns2=columns[9:]
        indexes=outliers(df[columns])
        new_df=df[~df.isin(indexes)]
        subplots(new_df,columns1)
        subplots(new_df,columns2)
        #endregion
        
        #region pairplot
        
        #endregion

        #Mapa de calor
        correlations=df.corr()
        fig=plt.figure(figsize=(12 ,12))
        ax=sns.heatmap(correlations,vmax=1,vmin=-1,square=True,cmap='viridis',annot=True,fmt=".2f",annot_kws={"size": 7})
        plt.title('Correlación de las variables')
        plt.show()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error_log(exc_tb.tb_lineno,e,'graficar')

def outliers(data: pd.DataFrame) ->List[str]:
    anomalies=[]
    #El objeto data es un df con columnas population y area, las funciones de mean y std y todas
    #las operaciones que hagamos con ellas nos regresarán una tupla/series con 2 valores
    mean=data.mean() #Objeto series
    std=data.std() #Objeto series
    limit_cut_off=std*2 #Un outlier se declara cuando sobrepasa dos veces la desviación estándar
    upper_limit= mean + limit_cut_off
    lower_limit= mean- limit_cut_off
    for index,row in data.iterrows():
        out=row
        #print(out)
        if out.iloc[0]>upper_limit.iloc[0] or out.iloc[0]<lower_limit.iloc[0]:
            anomalies.append(index)
    return anomalies

def main():
    root=os.path.dirname(os.path.realpath(__file__))
    df=pd.read_csv(os.path.join(root,'Facebook_metrics','dataset_Facebook.csv'),sep=';',dtype={'Type':'category'})
    #Nota: LP significa lifetime post, se cambió para simplificar los nombres de las características
    #así como se modificó el nombre de otras
    new_columns=['Page total likes', 'Type', 'Category', 'Post Month', 'Post Weekday',
       'Post Hour', 'Paid', 'LP Total Reach', 'LP Total Impressions', 'L Engaged Users', 'LP Consumers', 'LP Consumptions',
       'LP Impressions by people liked Page', 'LP reach by people like Page', 'LP liked Page andengaged post',
       'comment', 'like', 'share', 'Total Interactions']
    df.columns=new_columns
    descripcion(df)
    #graficar(df)

if __name__ =='__main__':
    logging.basicConfig(level=logging.INFO,filename='reporte.log')
    logging.info('\n\n***** Iniciando ejecución *****')
    logging.info('\nHora y fecha de ejecución: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    logging.info('\nHora y fecha de terminación: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('\n\n***** Terminando la ejecución *****\n')
