# Data Understanding--------------------------------------------------
## Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn import metrics

## Set display settings
pd.set_option('display.max_colwidth', 80)

## Change the default path
path='./data'
os.chdir(path)

## Read the datasets
user_features=pd.read_csv("user_features.csv")
product_features=pd.read_csv("product_features.csv")
click_history=pd.read_csv("click_history.csv")

## Explore "user_features":
### Print the shape 
print("\'user_features\':",user_features.shape )

### Which data_types does "user_features" contain?
user_features.dtypes

### What does "user_features" dataset look like?
user_features.head()

### How many unique value does each column of "user_features" have?
user_features.nunique()

### Does column "personal_interest" contain any empty list? 
(user_features.personal_interests.astype('str')=='[]').sum()

### How many douplicated "user_id" there is in "user_features"?
user_features.user_id.duplicated().sum()

### How many missing values there are in "user_features"?
user_features.isnull().sum()

### View the unique values of "number_of_clicks_before':
user_features.number_of_clicks_before.unique()

### Data exploration summary for "user_features": There is no duplicated row; there is no duplicated user_id; there are some missing values in "number_of_clicks_before". Also, column "personal_intrest" has some empty list.

## Explore "product_features"
print("\'product_features\':",product_features.shape)

### What data types does "product_features" dataset have?
product_features.dtypes

### How many unique values does each column have?
product_features.nunique()

### How does "product_features" look like?
product_features.head()

### Is there any duplicated "product_id"?
product_features.product_id.duplicated().sum()

### Is there any duplicated rows?
product_features.product_id.duplicated().sum()

### How many missing values?
product_features.isnull().sum()

product_features.describe()
product_features[["number_of_reviews","avg_review_score"]].hist()
fig, axs = plt.subplots(ncols = 2,figsize=(6,4))
plt.tight_layout(w_pad = 1.5)
product_features[["number_of_reviews"]].boxplot(ax = axs[0])
product_features[["avg_review_score"]].boxplot(ax = axs[1])

### Data exploration summary for "product_features": There is no duplicated row; there is no duplicated product_id; there is no missing values. The box plot for "number of reviews" shows there are some outliers. Also, avg_review_score is negetive for some data points.

### Explore "click_history"
print("\'click_history\':",click_history.shape)

### Which data types does "click_history" contain?
click_history.dtypes

### See the head of "click_history":
click_history.head()

### Is there any missing values?
click_history.isnull().sum()

### Is there any douplicated row in the "click_history" dataset?
click_history.duplicated().sum()

### Is there any douplicated row in the "click_history" dataset?
click_history.duplicated().sum()

### Data exploration summary for "click_history": It has 3 features, "user_id" , "product_id" and "clicked". Clicked is Boolean and it would be the target variable. Also, this data set does not have any duplicated rows, duplicated product_id or missing values.

# Data Cleaning and Preprocessing----------------------------------------------

## Remove negetive values of "product_features.avg_review_score"
lab=product_features[product_features.avg_review_score <0].index
product_features.drop(lab, axis=0,inplace=True)

##Merging


### Merge data sets
merged_data=click_history.merge(product_features,on="product_id",how="inner").merge(user_features,on="user_id",how="inner")
merged_data.shape

merged_data.head(5)

merged_data.duplicated().sum()

### Character replacement
merged_data.personal_interests=merged_data.personal_interests.str.replace('\[','')
merged_data.personal_interests=merged_data.personal_interests.str.replace('\]','')
merged_data.personal_interests=merged_data.personal_interests.str.replace('\'','')
merged_data.personal_interests=merged_data.personal_interests.str.replace(',','')


### View the head after the latest change
merged_data.head(3)


### Split the column of "personal_interests" :
merged_data = merged_data.join(pd.DataFrame(merged_data['personal_interests'].str.split().values.tolist()))
### Drop "personal_interests":
merged_data.drop(['personal_interests'],axis=1,inplace=True)
merged_data.shape
merged_data.head(3)

### The aim of this part is to make a dataframe so that all columns of 0 to 9 of merged_data_2 be stacked in one column.

df0=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",0]]
df0.rename(columns={0:'personal_interests'},inplace=True)

df1=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",1]]
df1.rename(columns={1:'personal_interests'},inplace=True)

df2=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",2]]
df2.rename(columns={2:'personal_interests'},inplace=True)

df3=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",3]]
df3.rename(columns={3:'personal_interests'},inplace=True)

df4=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",4]]
df4.rename(columns={4:'personal_interests'},inplace=True)

df5=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",5]]
df5.rename(columns={5:'personal_interests'},inplace=True)

df6=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",6]]
df6.rename(columns={6:'personal_interests'},inplace=True)

df7=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",7]]
df7.rename(columns={7:'personal_interests'},inplace=True)

df8=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",8]]
df8.rename(columns={8:'personal_interests'},inplace=True)

df9=merged_data[["clicked","product_id","category","on_sale","number_of_reviews","avg_review_score",
                   "user_id","number_of_clicks_before","ordered_before",9]]
df9.rename(columns={9:'personal_interests'},inplace=True)

mydata=pd.concat([df0,df1,df2,df3,df4,df5,df6,df7,df8,df9])
mydata.drop_duplicates(inplace=True)
mydata.dropna(subset=['personal_interests'],inplace=True)
mydata.shape
mydata.nunique()
mydata.isnull().sum()

### Drop the rows for them "number_of_clicks_before" is missing.
mydata.dropna(subset=['number_of_clicks_before'],inplace=True)

### Convert the categoric and boolean variables to numeric
for col in mydata.columns:
    if mydata[col].dtype in [bool,object] :
        mydata[col]=mydata[col].astype('category').cat.codes       

print("shape of 'mydata':",mydata.shape)
mydata.head()

### Is the target variable well_balanced?
mydata.clicked.hist()

### It seems that we have more target variables labeled as zero than one. So we are dealing with an imbalanced data.

# Model Generation and Evaluation----------------------------------------------
y=mydata.iloc[:,0]
X=mydata.iloc[:,1:10]

X.head()

## Split to train and test
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

## Rescale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Logistic regression
## The three lists will hold the best score, classifier name and the corresponding parameters.
## At the end, I will make a data frame and will find the best classifier. 
## Models comparison is based on AUC_ROC score of the test dataset.
best_scores=[]
best_models=[]
best_params=[]

grid_param={               
            'solver' :['lbfgs', 'liblinear', 'sag', 'saga'], 
            'class_weight':['balanced',None],
            'C':[0.00001,0.0001,0.001,0.01]   
            }   

best_score=0
c_train_score=0
parameters=[]
lr_test_auc_scores=[] 

for p in ParameterGrid(grid_param):
    clf= linear_model.LogisticRegression()
    clf.set_params(**p)

    parameters.append(p)
    fitted=clf.fit(X_train_scaled,y_train)
    
    lr_train_pred=fitted.predict(X_train_scaled)
    lr_test_pred=fitted.predict(X_test_scaled)
    
    lr_train_auc = metrics.roc_auc_score(y_train,lr_train_pred)
    lr_test_auc = metrics.roc_auc_score(y_test,lr_test_pred)
    lr_test_auc_scores.append(lr_test_auc)

    if lr_test_auc > best_score:
            best_param=p
            best_score=lr_test_auc
            c_train_score=lr_train_auc
print (best_param, ", best score:","{:.2%}".format(best_score),' ,correspinding train score:',"{:.2%}".format(c_train_score))

best_scores.append(best_score)
best_models.append("Logestic Regression")
best_params.append(best_param)


## Decision Tree
grid_param={'criterion':['gini','entropy'],
            'max_depth':range(3,20)
            }   

best_score=0
c_train_score=0
parameters=[]
dt_test_auc_scores=[]

for p in ParameterGrid(grid_param):
    clf= tree.DecisionTreeClassifier()
    clf.set_params(**p)

    parameters.append(p)
    fitted=clf.fit(X_train_scaled,y_train)
    
    dt_train_pred=fitted.predict(X_train_scaled)
    dt_test_pred=fitted.predict(X_test_scaled)
    
    dt_train_auc = metrics.roc_auc_score(y_train,dt_train_pred)
    dt_test_auc = metrics.roc_auc_score(y_test,dt_test_pred)
    dt_test_auc_scores.append(dt_test_auc)

    if dt_test_auc > best_score:
            best_param=p
            c_train_score=dt_train_auc
            best_score=dt_test_auc            
print (best_param, ", best test score:","{:.2%}".format(best_score),' ,correspinding train score:',"{:.2%}".format(c_train_score))
best_scores.append(best_score)
best_models.append("Decision Tree")
best_params.append(best_param)


## Naive Bayes
   
clf= GaussianNB(priors=None, var_smoothing=1e-09)

clf.fit(X_train_scaled,y_train)

nb_train_pred=fitted.predict(X_train_scaled)
nb_test_pred=clf.predict(X_test_scaled)

nb_test_auc = metrics.roc_auc_score(y_test,nb_test_pred)
nb_train_auc = metrics.roc_auc_score(y_train,nb_train_pred)

print("test score","{:.2%}".format(nb_test_auc),' ,correspinding train score:',"{:.2%}".format(nb_train_auc))
best_scores.append(nb_test_auc)
best_models.append("GaussianNB")
best_params.append({'priors':None, 'var_smoothing':1e-09})


## Neural Networks
# rescale data to [0, 1]
scaler = MinMaxScaler()
X_train_MMscaled = scaler.fit_transform(X_train)
X_test_MMscaled = scaler.transform(X_test)

grid_param={
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'solver':['adam'],
            'hidden_layer_sizes': [10,20,30,(5,5,5),(40,20)],
            'alpha': [0.0001,0.001,0.01,1,10],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'max_iter':[1000],
}

param_list = list(ParameterSampler(grid_param, n_iter=10))

best_score=0
c_train_score=0
parameters=[]
nn_test_auc_scores=[]
 
for p in param_list:
    clf=MLPClassifier()
    clf.set_params(**p)   
    parameters.append(p)
    fitted=clf.fit(X_train_MMscaled,y_train)
    
    nn_train_pred=fitted.predict(X_train_MMscaled)
    nn_test_pred=fitted.predict(X_test_MMscaled)
    nn_train_auc = metrics.roc_auc_score(y_train,nn_train_pred)
    nn_test_auc = metrics.roc_auc_score(y_test,nn_test_pred)
    nn_test_auc_scores.append(nn_test_auc)

    if nn_test_auc > best_score:
            best_param=p
            best_score=nn_test_auc
            c_train_score=nn_train_auc
print (best_param, ", best score:","{:.2%}".format(best_score),', correspinding train score:',"{:.2%}".format(c_train_score))
best_scores.append(best_score)
best_models.append("Neural Networks")
best_params.append(best_param)


## Random Forest
grid_param={
            'n_estimators':[150,200,250],
            'max_features':['auto','log2'],
            'max_depth':[3,10,15,20]   
            }   

best_score=0
c_train_score=0
parameters=[]
rf_test_auc_scores=[]
   
for p in ParameterGrid(grid_param):
    clf= RandomForestClassifier(n_jobs=-1)
    clf.set_params(**p)

    parameters.append(p)
    fitted=clf.fit(X_train_scaled,y_train)
    rf_train_pred=fitted.predict(X_train_scaled)
    rf_test_pred=fitted.predict(X_test_scaled)
    
    rf_train_auc = metrics.roc_auc_score(y_train,rf_train_pred)
    rf_test_auc = metrics.roc_auc_score(y_test,rf_test_pred)
    rf_test_auc_scores.append(rf_test_auc)

    if rf_test_auc > best_score:
            best_param=p
            best_score=rf_test_auc
            c_train_score=rf_train_auc
print (best_param, ", best score:","{:.2%}".format(best_score),', correspinding train score:',"{:.2%}".format(c_train_score))
best_scores.append(best_score)
best_models.append("Random Forest")
best_params.append(best_param)


## Support Vector Machine
grid_param={
            'kernel':['linear', 'poly', 'rbf'], 
            'degree' : [2,3]
            }   

best_score=0
c_train_score=0
parameters=[]
svm_test_auc_scores=[]
   
for p in ParameterGrid(grid_param):
    clf = svm.SVC(gamma='auto',max_iter=-1,class_weight='balanced')
    clf.set_params(**p)

    parameters.append(p)
    fitted=clf.fit(X_train_scaled,y_train)
    svm_train_pred=fitted.predict(X_train_scaled)
    svm_test_pred=fitted.predict(X_test_scaled)
    
    svm_train_auc = metrics.roc_auc_score(y_train,svm_train_pred)
    svm_test_auc = metrics.roc_auc_score(y_test,svm_test_pred)
    svm_test_auc_scores.append(svm_test_auc)

    if svm_test_auc > best_score:
            best_param=p
            best_score=svm_test_auc
            c_train_score=svm_train_auc
print (best_param, ", best score:","{:.2%}".format(best_score),', correspinding train score:',"{:.2%}".format(c_train_score))
best_scores.append(best_score)
best_models.append("SVM")
best_params.append(best_param)


## Ada Boosting
grid_param={
            'n_estimators':[50, 100, 150], 
            'learning_rate' : [0.0001,0.001,0.001,0.1,1],
            }   

best_score=0
c_train_score=0
parameters=[]
adb_test_auc_scores=[]
   
for p in ParameterGrid(grid_param):
    clf = AdaBoostClassifier()
    clf.set_params(**p)

    parameters.append(p)
    fitted=clf.fit(X_train_scaled,y_train)
    adb_train_pred=fitted.predict(X_train_scaled)
    adb_test_pred=fitted.predict(X_test_scaled)
    
    adb_train_auc = metrics.roc_auc_score(y_train,adb_train_pred)
    adb_test_auc = metrics.roc_auc_score(y_test,adb_test_pred)
    adb_test_auc_scores.append(adb_test_auc)

    if adb_test_auc > best_score:
            best_param=p
            best_score=adb_test_auc
            c_train_score=adb_train_auc
print (best_param, ", best score:","{:.2%}".format(best_score),', correspinding train score:',"{:.2%}".format(c_train_score))
best_scores.append(best_score)
best_models.append("Adb")
best_params.append(best_param)

# Finding the best model
# Make a dataframe which contains models name as index and the corresponding best score and parameters.
result=pd.DataFrame({"Test_Score":best_scores,"Param":best_params},index=best_models)
result=result.sort_values("Test_Score", ascending = True)


plt.barh(result.index,result.Test_Score)

m=max(result.Test_Score)
result[result.Test_Score==m]

# re-build the model with the best classifier and the best parameters
clf= RandomForestClassifier(n_jobs=-1,max_depth= 20, max_features= 'log2', n_estimators= 250)
clf.fit(X_train_scaled,y_train)
clf_y_test_pred=clf.predict(X_test_scaled)
print("roc_auc_score:","{:.2%}".format(metrics.roc_auc_score(y_test,clf_y_test_pred)))
print("accuracy:","{:.2%}".format(metrics.accuracy_score(y_test,clf_y_test_pred)))
print("precision:","{:.2%}".format(metrics.precision_score(y_test,clf_y_test_pred)))
print("recall:","{:.2%}".format(metrics.recall_score(y_test,clf_y_test_pred)))
print("f1:","{:.2%}".format(metrics.f1_score(y_test,clf_y_test_pred)))
print(metrics.confusion_matrix(y_test,clf_y_test_pred))


features=X.columns 
importances = clf.feature_importances_  

#Build a dataframe form features and the corresponding importance 
feature_importance=pd.DataFrame(importances,index=features,columns=['importance']).sort_values('importance',ascending=True)

#Draw feature importance bar plot
plt.figure(figsize=(6,5))
plt.title('Feature Importances')
plt.barh(feature_importance.index ,feature_importance.importance, color='green', align='center',height=0.5)
plt.xlabel('Relative Importance')
plt.show()


# Random Forest has the highest roc_auc score for the test dataset. "Number of reviwes" and "user_id" were the most important features respectively. 
# Data understanding and data cleansing was the hardest, the most time-consuming and the most important part. The quality of this part can greatly affect the quality of rest of work. 
# Amongst different classifiers, SVM was the slowest. 
