import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,StandardScaler, OrdinalEncoder 
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso,ElasticNet,Ridge,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import  RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import KFold, cross_val_score,train_test_split
import sys

def main():
    try:
        root=os.path.dirname(os.path.realpath(__file__))
        df=pd.read_csv(os.path.join(root,'Facebook_metrics','dataset_Facebook.csv'),sep=';',dtype={'Type':'category'})
        new_columns=['Page total likes', 'Type', 'Category', 'Post Month', 'Post Weekday',
        'Post Hour', 'Paid', 'LP Total Reach', 'LP Total Impressions', 'L Engaged Users', 'LP Consumers', 'LP Consumptions',
        'LP Impressions by people liked Page', 'LP reach by people like Page', 'LP liked Page andengaged post',
        'comment', 'like', 'share', 'Total Interactions']
        df.columns=new_columns
        values=df.values
        X=values[:,0:18]  
        Y=values[:,18]

        numerical=df.select_dtypes(include=['int64','float64']).columns
        categorical=df.select_dtypes(include=['object','bool','category']).columns

        #Pipeline para datos numéricos
        num_pipe=Pipeline([('imputer',SimpleImputer(strategy='mean')),
                            ('scaler',StandardScaler())])
        #Pipeline para datos de categoria
        cat_pipe=Pipeline([('Ordinal',OrdinalEncoder()),
                            ('OHE',OneHotEncoder())])
        full_pipe=ColumnTransformer([
                ('num',num_pipe,numerical.drop('Total Interactions')),
                ('cat',cat_pipe,categorical)
            ])
        cat_df=df.drop('Total Interactions',axis=1)
        X=full_pipe.fit_transform(cat_df)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=7)


        models=[('LASSO',Lasso()),
                ('ElasticNet',ElasticNet()),
                ('Ridge',Ridge()),
                ('Linear',LinearRegression()),
                ('SVR',SVR(kernel='poly',degree=2)),
                ('RFR',RandomForestRegressor(n_estimators=20,max_depth=3))]
            
        kfold=KFold(n_splits=10,random_state=7,shuffle=True)
        scoring='neg_mean_absolute_error'
        for name,model in models:
            pipe=Pipeline(steps=[
                                ('KBest',SelectKBest(k=8)),
                                #('Power',PowerTransformer(method='yeo-johnson')),
                                (name,model)
                        ])
        
            results=cross_val_score(pipe,X_train,Y_train,scoring=scoring,cv=kfold)
            print(f'Modelo:{name}\nNegative MAE: {results.mean():,.2f} Var: ({results.std():,.2f})')
    
        
    except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f'Ha ocurrido un eror en la línea {exc_tb.tb_lineno}: \n{e}')  

if __name__=='__main__':
    main()
