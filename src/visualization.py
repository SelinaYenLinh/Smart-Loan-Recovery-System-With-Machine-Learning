import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objs as go


pd.set_option('display.max_columns', None)
df = pd.read_csv(r"D:\Khôi\Khoá Học\ML Project\#01 - Machine Learning Projects On Real Business Problems\Smart Loan Recovery System With Machine Learning\data\loan-recovery.csv")

df = df.drop(columns= ['Borrower_ID', 'Loan_ID'])
# plt.figure(figsize=(10,3))
# sns.heatmap(df.corr(numeric_only= True), annot= True, cmap= 'coolwarm', fmt= '.2f')
# plt.show()
# # print(df.corr(numeric_only= True))
file_path = r"D:\Khôi\Khoá Học\ML Project\#01 - Machine Learning Projects On Real Business Problems\Smart Loan Recovery System With Machine Learning\chart"
fig = px.histogram(data_frame= df, 
                   x = 'Loan_Amount',
                   marginal= 'violin', 
                   opacity= 0.2, 
                   histnorm= '',
                   title= 'Loan Amount Distribution & Relationship with Month Income',
                   color_discrete_sequence=["royalblue"])

fig.add_trace(go.Scatter(x = sorted(df['Loan_Amount']), 
                        y = px.histogram(df, x = 'Loan_Amount', nbins= 30, histnorm= 'probability').data[0].y,
                        mode= 'lines',
                        name= 'Density',
                        line= dict(color='red', width = 2, dash = 'solid')))
scatter = px.scatter(data_frame= df, 
                     x = 'Loan_Amount', 
                     y= 'Monthly_Income', 
                     size= 'Loan_Amount', 
                     color= 'Loan_Amount')
for trace in scatter.data:
    fig.add_trace(trace)
fig.update_layout(yaxis_title = 'Monthly Income (In $)',
                  xaxis_title = 'Loan Amount (In $)',
                  font =  dict(color = 'black',
                              size = 12,
                              family = 'Arial'),
                  template="plotly_white"
                  )
fig.add_annotation(x= max(df['Loan_Amount'])*0.9,
                   y = max(df['Monthly_Income']),
                   text = 'Higher income lead to higher loan amount',
                   arrowhead= 2,
                   showarrow= True,
                   font= dict(color = 'red',
                              size = 12,
                              family = 'Arial'))
img_path = os.path.join(file_path, 'Loan Amount Distribution & Relationship with Month Income.png')
fig.write_image(img_path)

fig = px.histogram(data_frame= df,
                   x= 'Payment_History',
                   color= 'Recovery_Status',
                   barmode= 'group'
                   )
fig.update_layout(xaxis_title = 'Payment History',
                  yaxis_title = 'Number Of Loans',
                  font = dict(size = 12, 
                              family = 'Arial'))
img_path = os.path.join(file_path, 'How Payment History Affects Loan Recovery Status.png')
fig.write_image(img_path)

fig = px.box(data_frame= df, 
             x = 'Recovery_Status',
             y = 'Num_Missed_Payments',
             color= 'Recovery_Status', 
             points= 'all',
             title= "How Missed Payments Affect Loan Recovery Status")
fig.update_layout(xaxis_title = 'Recovery Status',
                  yaxis_title = 'Num Of Missed Payments',
                  font = dict(size = 12, 
                              family = 'Arial'),
                  template="plotly_white")

img_path = os.path.join(file_path, 'How Missed Payments Affect Loan Recovery Status.png')
fig.write_image(img_path)

fig = px.scatter(data_frame= df,
                 x = 'Monthly_Income',
                 y = 'Loan_Amount',
                 color= 'Recovery_Status',
                 size= 'Loan_Amount',
                 title= 'How Monthly Income and Loan Amount Affect Loan Recovery')
fig.add_annotation(x = max(df['Monthly_Income'])*0.9,
                   y = max(df['Loan_Amount']),
                   text= 'Higher loans may still get recovered if income is high',
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
img_path = os.path.join(file_path, 'How Monthly Income and Loan Amount Affect Loan Recovery.png')
fig.write_image(img_path)



