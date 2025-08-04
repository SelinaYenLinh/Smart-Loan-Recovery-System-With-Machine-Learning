import pandas as pd
import os
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from preprocessing import preprocessor
from sklearn.pipeline import Pipeline
from segmentation import segment
from model_training import riskclassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"D:\Khôi\Khoá Học\ML Project\#01 - Machine Learning Projects On Real Business Problems\Smart Loan Recovery System With Machine Learning\data\loan-recovery.csv")
df = df.drop(columns= ['Borrower_ID', 'Loan_ID'])
folder_path = 'D:\Khôi\Khoá Học\ML Project\#01 - Machine Learning Projects On Real Business Problems\Smart Loan Recovery System With Machine Learning'

numeric_features = ['Monthly_Income', 'Num_Dependents', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI', 'Num_Missed_Payments',
            'Days_Past_Due']
categorical_features = ['Gender', 'Employment_Type', 'Loan_Type', 
            'Payment_History', 'Collection_Method', 'Collection_Method']
ordinal_features = ['Recovery_Status']

x_train_raw, x_test_raw = train_test_split(df,test_size= 0.2, random_state= 42)
pre = preprocessor(numeric_features, categorical_features, ordinal_features)
x_train_process = pre.fit_transform(x_train_raw)
x_test_process = pre.transform(x_test_raw)

kmean = segment()
x_train_raw['cluster'] = kmean.fit_predict(x_train_process)
x_test_raw['cluster'] = kmean.predict(x_test_process)
x_train_raw['segment'] = kmean.map_segment_labels(x_train_raw['cluster'])
x_test_raw['segment'] = kmean.map_segment_labels(x_test_raw['cluster'])

y_train = x_train_raw['segment'].apply(lambda x : 1 if x in ['Moderate Income, High Loan Burden',
                                                                 'High Loan, Higher Default Risk'] else 0)
y_test = x_test_raw['segment'].apply(lambda x : 1 if x in ['Moderate Income, High Loan Burden',
                                                                 'High Loan, Higher Default Risk'] else 0)

# Draw chart by segments
df['Brower_segment'] = pd.concat([x_train_raw['segment'], x_test_raw['segment']], ignore_index= True)

fig = px.scatter(data_frame= df,
                 x = 'Monthly_Income',
                 y = 'Loan_Amount',
                 color= 'Brower_segment',
                 size= 'Loan_Amount',
                 title= 'Borrower Segments Based on Monthly Income and Loan Amount')
fig.add_annotation(x = max(df['Monthly_Income'])*0.5,
                   y = max(df['Loan_Amount']),
                   text= 'Higher loans are clustered in specific income groups',
                   arrowhead= 2,
                   showarrow= True,
                   font= dict(color = 'red',
                              size = 12,
                              family = 'Arial')
                   )
fig.update_layout(xaxis_title = 'Monthly Income',
                  yaxis_title = 'Loan Amount',
                  font = dict(size = 12, 
                              family = 'Arial'),
                  template="plotly_white",
                  xaxis = dict(tick0 = 0,
                              dtick = 50000,
                              tickformat = '~s'))
name = 'Borrower Segments Based on Monthly Income and Loan Amount.png'
save_chart_path = os.path.join(folder_path, 'chart')
os.makedirs(save_chart_path, exist_ok= True)
fig.write_image(os.path.join(save_chart_path, name))

model = riskclassifier()
model.train(x_train_process, y_train)

# print(model.evaluation(x_test_process, y_test))

save_folder_path = os.path.join(folder_path, 'artifacts')
os.makedirs(save_folder_path, exist_ok= True)
joblib.dump(model.model, os.path.join(save_folder_path, 'model.pkl'))
joblib.dump(pre.preprocessor, os.path.join(save_folder_path, 'preprocessor.pkl'))

features = numeric_features + ordinal_features + categorical_features
with open(os.path.join(save_folder_path, 'features.json'), 'w') as f:
    json.dump(features, f)

metadata = {
    'model_name': type(model.model).__name__,
    'date_train' : datetime.now().strftime('%y-%m-%d:%H:%M:%S'),
    'features': features,
    'segment_mapping': kmean.cluster_map,
    'note': 'segment 1 mapped to highed risk'
}
with open(os.path.join(save_folder_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)






