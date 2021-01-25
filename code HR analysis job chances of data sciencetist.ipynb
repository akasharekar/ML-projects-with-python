
import numpy as np
import pandas as pd

# ans test from kaggle 
test_result = np.load(r"E:\practice\HR analysis job chances of data sciencetist\jobchange_test_target_values.npy")
test_result=pd.DataFrame(test_result)

#import the dat set

train=pd.read_csv(r"E:\practice\HR analysis job chances of data sciencetist\aug_train.csv",header=0) 
test=pd.read_csv(r"E:\practice\HR analysis job chances of data sciencetist\aug_test.csv",header=0)

print(train.shape)
print(test.shape)

print(train.info())
print(test.info())

print(train.isnull().sum())
print(test.isnull().sum())

# check the unique values in each columns
train.columns

#for tarin data set
# alwasy use unique() function
for i in train.columns:
    print(i,train[i].unique())
    
# for test data set    
for i in test.columns:
    print(i,test[i].unique())

#train.enrollee_id.unique()
    
train.gender.unique()
train.gender.value_counts()  
    
# =============================================================================
# #fill na values with mode in "experience" column
# =============================================================================
# for train data
#work with "experience" column
# change unique values    
#">20"=21
#"<1"=0
       
train["experience"].replace([">20","<1"],["21","0"],inplace=True)

train["experience"].unique()

train["experience"].fillna(train["experience"].mode()[0],inplace=True)
train["experience"].unique()
train["experience"].isnull().sum()
train["experience"] = train["experience"].astype("int")

# for test data
test["experience"].replace(["<1",">20"],["0","21"],inplace=True)

test["experience"].unique()

test["experience"].fillna(test["experience"].mode()[0],inplace=True)
test["experience"].unique()
test["experience"].isnull().sum()
test["experience"] = test["experience"].astype("int")

train.info()
test.info()
# =============================================================================
# # work on  last_new_job column 
# =============================================================================
# for train data
train["last_new_job"].unique()
 
train["last_new_job"].replace([">4","never"],["5","0"],inplace=True)
 
train["last_new_job"].fillna(int(train["last_new_job"].mode()[0]),inplace=True)
train["last_new_job"].unique()
train["last_new_job"] = train["last_new_job"].astype("int")
 
#for test data 
test["last_new_job"].unique()
 
test["last_new_job"].replace([">4","never"],["5","0"],inplace=True)
 
test["last_new_job"].fillna(int(train["last_new_job"].mode()[0]),inplace=True)
test["last_new_job"].unique()
test["last_new_job"] = test["last_new_job"].astype("int")


# =============================================================================
# # work on "company_size"  column 
# =============================================================================
train["company_size"].unique()
train["company_size"].value_counts()

print(train["company_size"].mode())
train["company_size"].replace(["<10","10000+","10/49"],["0-10","10000","10-49"],inplace=True)

train["company_size"].fillna(train["company_size"].mode()[0],inplace=True)

#for test data
test["company_size"].unique()
train["company_size"].value_counts(dropna=False)

test["company_size"].replace(["<10","10000+","10/49"],["0-10","10000","10-49"],inplace=True)

test["company_size"].fillna(test["company_size"].mode()[0],inplace=True)


# =============================================================================
# # fill na values for other columns
# =============================================================================
#Index([  'gender',
#       , 'enrolled_university', 'education_level',
#       'major_discipline', 'company_type'])
      
train.isnull().sum()
test.isnull().sum()


#for x in train.columns:
#    if (train[x].dtype=="object" and train[x].isnull==True):
#        print(train[i].fillna(int(train[i]).mode()[0]),inplace=True)
        
        
train["gender"].fillna(train["gender"].mode()[0],inplace=True)
train["enrolled_university"].fillna(train["enrolled_university"].mode()[0],inplace=True)
train["education_level"].fillna(train["education_level"].mode()[0],inplace=True)
train["major_discipline"].fillna(train["major_discipline"].mode()[0],inplace=True)
train["company_type"].fillna(train["company_type"].mode()[0],inplace=True)

# for test data
test["gender"].fillna(test["gender"].mode()[0],inplace=True)
test["enrolled_university"].fillna(test["enrolled_university"].mode()[0],inplace=True)
test["education_level"].fillna(test["education_level"].mode()[0],inplace=True)
test["major_discipline"].fillna(test["major_discipline"].mode()[0],inplace=True)
test["company_type"].fillna(test["company_type"].mode()[0],inplace=True)


# =============================================================================
#  separate data X into Y
# =============================================================================
train.columns
X=train.iloc[:,0:13]
X.columns
Y=train.iloc[:,-1]
#Y.columns


X.info()



# =============================================================================
# # running lebel encoder on oject data 
# =============================================================================
col=[]
for r in X.columns:
    if X[r].dtype=="object":
        col.append(r)
print(col)


from sklearn import preprocessing

le=preprocessing.LabelEncoder()


for x in col:
    X[x]=le.fit_transform(X[x])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print('Feature', x)
    print('mapping', le_name_mapping)

# perform leble encoder on test data
for a in col:
    test[a]=le.fit_transform(test[a])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print('Feature', x)
    print('mapping', le_name_mapping)
    
test.info() 
# =============================================================================
# use standard scaler
# =============================================================================

from sklearn.preprocessing import StandardScaler 
# scale only X array not Y array.

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
# scalling for test data
test=scaler.transform(test)
print(X)
    
    
# =============================================================================
# # split the data into traing and testing part
# =============================================================================
    
# split the data into train and test
from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=10)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# =============================================================================
# # run the model (logistic regg)
# =============================================================================

from sklearn.linear_model import LogisticRegression
 
# create a model
classifier=LogisticRegression()

#fitting traing data to the model
classifier.fit(X_train,Y_train)

# predict using the model
Y_pred=classifier.predict(X_test)# what are Y values for given X values

# =============================================================================
# # make the confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("classification report: ")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("acccurecy of the model: ",acc)



# =============================================================================
# # predict on testing data 
# =============================================================================
test.shape
test_pred=classifier.predict(test)

test_pred=pd.DataFrame(test_pred)

acc = accuracy_score(test_result,test_pred)
print("acccurecy of the model: ",acc)

# =============================================================================
# # adjusting the threshold
# =============================================================================

# store the predicted probabilities
Y_pred_prob=classifier.predict_proba(X_test)# defalt threshold is 0.5
print(Y_pred_prob)


Y_pred_class=[]
for value in Y_pred_prob[:,1]: #second col [:,1] 
    if value > 0.47:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)

#print(Y_pred_class)        

###check the confution matrix#

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred_class)
print(cfm)

print("classification report: ")
print(classification_report(Y_test,Y_pred_class))

acc = accuracy_score(Y_test,Y_pred_class)
print("acccurecy of the model: ",acc)

# =============================================================================
# # predict on testing data (tuning part ) 
# =============================================================================
test.shape
Y_pred_prob_test=classifier.predict_proba(test)# defalt threshold is 0.5
print(Y_pred_prob_test)

#test_pred1=pd.DataFrame(Y_pred_prob_test)

Y_pred_class1=[]
for value in Y_pred_prob_test[:,1]: #second col [:,1] 
    if value > 0.42:
        Y_pred_class1.append(1)
    else:
        Y_pred_class1.append(0)

cfm=confusion_matrix(test_result,Y_pred_class1)
print(cfm)

print("classification report: ")
print(classification_report(test_result,Y_pred_class1))


acc = accuracy_score(test_result,Y_pred_class1)
print("acccurecy of the model: ",acc)







