from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def X_data_preprocessing(data):
    col_list = [0, 32, 39, 50, 54, 59, 65, 71, 76, 113, 116, 120]
    for x in col_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [x])], remainder='passthrough')
        data = ct.fit_transform(data)
    return data