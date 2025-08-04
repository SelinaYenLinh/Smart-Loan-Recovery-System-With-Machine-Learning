import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class preprocessor:
    def __init__(self, numeric_features, categorical_features, ordinal_features):
        self.numeric_features = numeric_features   
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features

        self.preprocessor = ColumnTransformer(
                            transformers= [
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown= 'ignore'), categorical_features),
                            ('ord', OrdinalEncoder(), ordinal_features)
                        ])
    def fit_transform(self, df):
        return self.preprocessor.fit_transform(df)
    def transform(self, df):
        return self.preprocessor.transform(df)
