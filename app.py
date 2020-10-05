import streamlit as st
import os  #importing OS to get file path
import numpy as np #To perform Liner Algebra
import pandas as pd #importing Pandas to have liverty to perform many tasks
import matplotlib.pyplot as plt #To plot charts
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer # Creating TD-IDF 
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix # create Confusion Matrix, Accuracy
from sklearn.feature_selection import chi2 #Feature Selection
from sklearn.model_selection import cross_val_score #getting Cross Validation Score
from sklearn.model_selection import train_test_split #Spliting Data To Test And Train Data
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns #For Plotting Graphs
from sklearn.pipeline import Pipeline #to work list of function as single iteration
from sklearn.linear_model import SGDClassifier

path = str(os.path.dirname(os.path.abspath(__file__))).replace('\\','/')

file = str(path+'/rows.csv')
@st.cache
def load_data(allow_output_mutation=True):
    df = pd.read_csv(file,low_memory=False)
    return df

df=load_data()
shap = df.shape
rows,col = shap
#st.write(os.getcwd())
st.title("Customer Complain Prediction into Pre-defined Groups")

st.write("In dataset there are ",rows," Rows and ",col," Columns")

columns = list(df.columns)
st.write("List Of Columns In DataSet Are As Follows")
st.table(columns)

st.write("Glimpse of Data In Given DataSet")
st.table(df.head(2).T)

#getting Null Column Count
column_null_count = df.isnull().sum().tolist()
null_col = 0
for cnc in column_null_count:
    if cnc!=0:
        null_col=null_col+1

st.write("Out of",col," Columns",null_col," Columns contains Null Values\n")

#st.write("Timely response? Unique:", df["Timely response?"].unique().tolist())
@st.cache
def get_nullcol():
    nul_col = pd.DataFrame(columns = ['Column Name', 'Null Count', 'Percentage'])
    total_rows = len(df)
    col_name=[]
    null_cnt=[]
    percent=[]
    for i in range(len(columns)):
        if(column_null_count[i]!=0):
            per = float((column_null_count[i]/total_rows)*100)
            col_name.append(columns[i])
            null_cnt.append(column_null_count[i])
            percent.append("%.2f" %per)
    nul_col["Column Name"] = col_name
    nul_col["Null Count"] = null_cnt
    nul_col["Percentage"] = percent
    return nul_col
    
nul_col = get_nullcol()
st.write("Displaying Null Count Of Columns")
st.table(nul_col)

st.write("Displaying Distinct Product Count (Before Converting)")

prod=[]
cnt=[]
for p in df['Product'].unique().tolist():
    tst = df[df['Product']==p]
    rw,cl = tst.shape
    prod.append(p)
    cnt.append(rw)
dict_pro = pd.DataFrame(index= prod,columns = ['Count'])
dict_pro1 = pd.DataFrame(columns = ['Product','Count'])
dict_pro1['Product'] = prod
dict_pro1['Count'] = cnt
dict_pro['Count'] = cnt

#st.table(dict_pro)
st.area_chart(dict_pro)

dict_pro1.replace({'Product': 
             {'Virtual currency' : 'Money transfer, virtual currency, or money service',
             'Money transfers' : 'Money transfer, virtual currency, or money service',
             'Prepaid card' : 'Credit card or prepaid card',
             'Credit card' : 'Credit card or prepaid card',
             'Payday loan, title loan, or personal loan' : 'Other Product',
             'Payday loan' : 'Other Product',
             'Credit reporting' : 'Credit reporting, credit repair services, or other personal consumer reports',
             'Other financial service' : 'Other Product'}}, inplace= True)


st.write("Displaying Products After Converting Some of products")

#Getting Unique Product Count
dict_pro1 = dict_pro1.groupby('Product').sum()
#st.table(dict_pro1)

dict_p1 = pd.DataFrame(index= dict_pro1.index.tolist(),columns = ['Count'])
dict_p1['Count'] = dict_pro1['Count'].tolist()
st.area_chart(dict_p1)
#---------
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(data=df,x='Product',hue='Timely response?')
plt.xticks(rotation=90)
fig.suptitle('Timely Response based On Products')
st.pyplot()

fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(data=df,x='Product',hue='Submitted via')
plt.xticks(rotation=90)
fig.suptitle('Complaints about Products Submitted Via')
st.pyplot()

#st.write(len(df['Company'].unique()))

fig, ax = plt.subplots(figsize=(8,6))
df.groupby('Product')['Issue'].count().plot.barh(ylim=0)
plt.xticks(rotation=90)
fig.suptitle('Issue Count Based On Product')
st.pyplot()

fig, ax = plt.subplots(figsize=(8,6))
df.groupby('Product')['Company'].count().plot.barh(ylim=0)
plt.xticks(rotation=90)
fig.suptitle('Product Complaint Count Based On Company')
st.pyplot()
#---------------

sp1=[]
cnt=[]
for sp in df['Sub-product'].unique().tolist():
    tst = df[df['Sub-product']==sp]
    rw,cl = tst.shape
    sp1.append(sp)
    cnt.append(rw)
dict_sp = pd.DataFrame(columns = ['Subproduct', 'Count'])

dict_sp['Subproduct'] = sp1
dict_sp['Count'] = cnt

other_sp=[]
keep_sp=[]
keep_sp_cnt=[]
other_cnt=0
for i in range(0,len(dict_sp)):
    if int(dict_sp.Count[i]) < 10000:
        #st.write(i," ",dict_sp.Subproduct[i])
        other_sp.append(dict_sp.Subproduct[i])
        other_cnt = other_cnt + dict_sp.Count[i]
    else:
        keep_sp.append(dict_sp.Subproduct[i])
        keep_sp_cnt.append(dict_sp.Count[i])

st.write(dict_sp)

st.write("Displaying Sub Products Before Converting")
dictsp1 = pd.DataFrame(index=sp1, columns = ['Count'])
dictsp1['Count'] = cnt
st.bar_chart(dictsp1)

keep_sp.append('Other')
keep_sp_cnt.append(other_cnt)

#st.write("Mean of Sub Product Count %.2f" %dict_sp.Count.mean())
st.write("There were total ",len(other_sp)," Sub Products Whose Count was less then 10,000")
new_sp = pd.DataFrame(index = keep_sp,columns = ['Count'])
#new_sp['Subproduct'] = keep_sp
new_sp['Count'] = keep_sp_cnt
#st.write(new_sp)

st.write("Diaplaying Sub Products After Converting")
st.bar_chart(new_sp)

issue = []
cnt=[]
for i in df['Issue'].unique().tolist():
    tst = df[df['Issue']==i]
    rw,cl = tst.shape
    issue.append(i)
    cnt.append(rw)
dict_issue = pd.DataFrame(columns = ['Issue', 'Count'])

dict_issue['Issue'] = issue
dict_issue['Count'] = cnt


other_isu=[]
keep_isu=[]
keep_isu_cnt=[]
other_cnt=0
for i in range(0,len(dict_issue)):
    if int(dict_issue.Count[i]) < 10000:
        other_isu.append(dict_issue.Issue[i])
        other_cnt = other_cnt + dict_issue.Count[i]
    else:
        keep_isu.append(dict_issue.Issue[i])
        keep_isu_cnt.append(dict_issue.Count[i])

keep_isu.append('Other')
keep_isu_cnt.append(other_cnt)

st.write("Out of ",len(dict_issue)," Issues, ", len(other_isu), " issues has count less then 10,000")

new_issue = pd.DataFrame(index = keep_isu,columns = ['Count'])
new_issue['Count'] = keep_isu_cnt

data = new_issue
names = list(keep_isu)
values = list(keep_isu_cnt)
fig, axs = plt.subplots()
plt.scatter(names,values)
plt.xticks(rotation=90)
fig.suptitle('Issue Count After converting')
st.pyplot()

df1 = df[['Product', 'Sub-product','Issue','Consumer complaint narrative']].copy()
df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]
#st.write(len(df1))


df1.replace({'Product': 
             {'Virtual currency' : 'Money transfer, virtual currency, or money service',
             'Money transfers' : 'Money transfer, virtual currency, or money service',
             'Prepaid card' : 'Credit card or prepaid card',
             'Credit card' : 'Credit card or prepaid card',
             'Payday loan, title loan, or personal loan' : 'Other Product',
             'Payday loan' : 'Other Product',
             'Credit reporting' : 'Credit reporting, credit repair services, or other personal consumer reports',
             'Other financial service' : 'Other Product'}}, inplace= True)

#st.write(df1['Product'].unique().tolist())
for osp  in other_sp:
    df1.replace({'Sub-product': 
             {osp : 'Other'}}, inplace= True)

#st.write(df1['Sub-product'].unique().tolist())

for isu  in other_isu:
    df1.replace({'Issue': 
             {isu : 'Other'}}, inplace= True)


st.write("Displaying Products After Removing All Null Values From Customer Complain Narrative")
p1=[]
cnt=[]
for p in df1['Product'].unique().tolist():
    tst = df1[df1['Product']==p]
    rw,cl = tst.shape
    p1.append(p)
    cnt.append(rw)
dict_p = pd.DataFrame(index= p1,columns = ['Count'])
dict_p['Count'] = cnt
st.bar_chart(dict_p)



df1['PID'] = df1['Product'].factorize()[0]
df1['SPID'] = df1['Sub-product'].factorize()[0]
df1['IID'] = df1['Issue'].factorize()[0]


sample_df= df1.sample(10000, random_state=0).copy()
sample_prod_cc = sample_df[['Product','Consumer complaint narrative','PID']].copy()
sample_pcc = dict(sample_df[['Product','PID']].copy().values)



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4,ngram_range=(1, 2),stop_words='english')

#Converting Customer Complaints to Vector Form
cc = tfidf.fit_transform(sample_df['Consumer complaint narrative']).toarray()

#st.write(cc.shape)
#temp = np.array(tfidf.inverse_transform(cc))
#st.write(temp)

lables = sample_df.PID
#st.write(tfidf)

st.write("For given Products which words occur together more frequently")
@st.cache
def get_unibigram():
    #For given Products which words occur together more frequently
    N = 5
    unigram=[]
    bigram=[]
    product=[]
    for Product, PID in sorted(sample_pcc.items()):
        cc_chi2 = chi2(cc, lables == PID)
        indices = np.argsort(cc_chi2[0])
        cc_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in cc_names if len(v.split(' ')) == 1]
        bigrams = [v for v in cc_names if len(v.split(' ')) == 2]
        product.append(Product)
        unigram.append(', '.join(unigrams[-N:]))
        bigram.append(', '.join(bigrams[-N:]))

    uni_bi_dict = pd.DataFrame(columns = ['Product','Unigrams','Bigrams'])
    uni_bi_dict['Product'] = product
    uni_bi_dict['Unigrams'] = unigram
    uni_bi_dict['Bigrams'] = bigram
    return uni_bi_dict

uni_bi_dict = get_unibigram()
st.table(uni_bi_dict)


X = sample_df['Consumer complaint narrative'] # Collection of documents
Y = sample_df['Product'] # Target Which we want to predict (different complaints of products)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25,random_state = 0)

modelname = 'Random Forest'
st.write(modelname)

@st.cache
def getrf():
    #Random Forest Classification    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    rf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', model),
                  ])
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy

acc = getrf()
st.write('Accuracy of Random Forest: ',(acc*100))

@st.cache
def getrfreport():
    #Random Forest Classification    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    # The pipeline can then be fit and applied to our dataset just like a single transform
    rf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', model),
                  ])
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    report = pd.DataFrame(metrics.classification_report(Y_test, y_pred,target_names=sample_df['Product'].unique(),output_dict=True)).transpose()
    return report

report = getrfreport()
st.table(report)

modelname = 'Naive Basian'
st.write(modelname)

@st.cache
def getNaiv():
    model = MultinomialNB()    
    accuracies = cross_val_score(model, cc, lables, scoring='accuracy', cv=5)
    return accuracies.mean()

nbacc = getNaiv()
st.write("Accuracy Of Naive Basian: ",(nbacc*100))

@st.cache
def getnbreport():   
    model = MultinomialNB()
    rf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', model),
                  ])
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    report = pd.DataFrame(metrics.classification_report(Y_test, y_pred,target_names=sample_df['Product'].unique(),output_dict=True)).transpose()
    return report

report = getnbreport()
st.table(report)


modelname = 'Linear SVM(SVC)'
st.write(modelname)
@st.cache
def getLSV():  
    model = LinearSVC()       
    accuracies = cross_val_score(model, cc, lables, scoring='accuracy', cv=5)
    return accuracies.mean()

lsvacc = getLSV()
st.write("Accuracy Of Linear SVM(SVC): %.2f" %(lsvacc*100))

@st.cache
def getlsvmreport():   
    model = LinearSVC()
    rf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', model),
                  ])
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    report = pd.DataFrame(metrics.classification_report(Y_test, y_pred,target_names=sample_df['Product'].unique(),output_dict=True)).transpose()
    return report

report = getlsvmreport()
st.table(report)

modelname = 'SGD ALGORITHM'
st.write(modelname)
@st.cache
def getSGD():
    #model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
    sgd.fit(X_train, Y_train)

    y_pred = sgd.predict(X_test)
    
    return accuracy_score(y_pred, Y_test)
    #accuracies = cross_val_score(model, cc, lables, scoring='accuracy', cv=3)
    #return accuracies.mean()

sgdacc = getSGD()
st.write("Accuracy Of SGD ALGORITHM: ",(sgdacc*100))

@st.cache
def getsgdreport():  
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
    sgd.fit(X_train, Y_train)

    y_pred = sgd.predict(X_test)
    
    report = pd.DataFrame(metrics.classification_report(Y_test, y_pred,target_names=sample_df['Product'].unique(),output_dict=True)).transpose()
    return report

report = getsgdreport()
st.table(report)

@st.cache
def plotConfMat():
    X_train, X_test, Y_train, Y_test = train_test_split(cc,lables, test_size=0.25,random_state=0)
    model = LinearSVC()
    model.fit(X_train, Y_train)
    Y_Predict = model.predict(X_test)
    conf_mat = confusion_matrix(Y_test, Y_Predict)
    return conf_mat
    
conf = plotConfMat();
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf, annot=True, cmap="Reds", fmt='d',
            xticklabels=sample_df['Product'].unique(), 
            yticklabels=sample_df['Product'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX \n")
st.pyplot()

#Predicting USer Input Complaints
x_vector = tfidf.fit(X_train)
x_vectors = x_vector.transform(X_train)
model = LinearSVC().fit(x_vectors, Y_train)

st.write("Lest Predict Complaints By Taking User Inpt")

user_input = st.text_input("Enter Your Question")
ans = ''
if user_input != "":
    ans = model.predict(x_vector.transform([user_input]))
    #print(user_input)
    st.write(ans)
