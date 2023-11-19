# Instructions
# Install the necessary packages
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import you data and perform basic data exploration phase
data = pd.read_csv('Expresso_churn_dataset.csv')
df = data.copy()

# Display general information about the dataset
df.info()
# Create a pandas profiling reports to gain insights into the dataset
# %pip install pandas pandas-profiling
# import pandas_profiling as pp
# from pandas_profiling import ProfileReport
# Handle Missing and corrupted values
# df.isnull().sum()
# df.drop('user_id', axis = 1, inplace = True)
# for i in df:
#     if df[i].isnull().sum() / len(df) * 100 > 50:
#         df.drop(i, axis = 1, inplace = True)
# sampling0 = df[df.CHURN == 0]
# sampling0 = sampling0.dropna()
# sampling0.reset_index(drop = True, inplace = True)
# sampling1 = df[df.CHURN == 1]
# sampling1.drop(['REGION', 'TOP_PACK', 'FREQ_TOP_PACK'], axis = 1, inplace = True)
# sampling1.dropna(inplace = True)
# sampling1.reset_index(drop = True, inplace =True)
# sampling0 = sampling0.sample(35000)
# cols = sampling1.columns
# dx = pd.concat([sampling1, sampling0[cols]], axis = 0)
# df = dx.copy()
# df.dropna()
# Remove duplicates, if they exist
df.duplicated()
all_duplicates = df[df.duplicated(keep=False)]
df_no_duplicates = df.drop_duplicates()
# Handle outliers, if they exist
def outlierRemoval(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtypes != 'O': # --------------------------------------- If the data type is not an object type
            Q1 = dataframe[i].describe()[4] # ---------------------------------- Identify lower Quartile
            Q3 = dataframe[i].describe()[6] # ---------------------------------- Identify the upper quartile
            IQR = Q3 - Q1 # ---------------------------------------------------- Get Inter Quartile Range
            minThreshold = Q1 - 1.5 * IQR # ------------------------------------ Get Minimum Threshold
            maxThreshold = Q3 + 1.5 * IQR # ------------------------------------ Get Maximum Threshold

            dataframe = dataframe.loc[(dataframe[i] >= minThreshold) & (dataframe[i] <= maxThreshold)]
    return dataframe


df = outlierRemoval(df)
# Encode categorical features
# Encode categorical features
categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in df.drop('CHURN', axis = 1).columns: # ...................................................... If the selected column is found in the general dataframe
        df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in df.drop('CHURN', axis = 1).columns: # ...................................................... If the selected columns are found in the general dataframe
        df[i] = encoder.fit_transform(df[i])# .................................. encode it

sel_cols = ['REGULARITY', 'DATA_VOLUME','REVENUE',  'ORANGE', 'ON_NET', 'MONTANT','FREQUENCE']
x = df[sel_cols]

x = x
y = df.CHURN
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 75, stratify = y)
# Based on the previous data exploration train and test a machine learning classifier
# Based on the previous data exploration train and test a machine learning classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# save model
model = pickle.dump(model, open('Expresso.pkl', 'wb'))

# Create a streamlit application (locally)
model = pickle.load(open('Expresso.pkl','rb'))

st.markdown("<h1 style = 'color: #EE9322; text-align: center;font-family: Arial, Helvetica, sans-serif; '>ESPRESSO TELECOMMUNICATIONS</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: ##EE9322; text-align: center;font-family: Arial, Helvetica, sans-serif; '> Created by Ayodeji</h3>", unsafe_allow_html= True)
st.image('Mast.png', width = 600)
st.markdown("<h2 style = 'color: #EE9322; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

# st.markdown('<br><br>', unsafe_allow_html= True)

st.markdown("<p>Espresso is a telecommunications brand operating under the Sudatel Group in Africa. With a presence in multiple countries, Expresso provides essential services such as mobile telephony and internet connectivity. As part of Sudatel, a significant player in African telecom, Expresso contributes to connectivity, economic development, and social interactions. The brand faces challenges but also has opportunities for innovation in the dynamic telecommunications landscape.</p>",unsafe_allow_html= True)

st.sidebar.image('USER.png')

dx = data[['REGULARITY', 'DATA_VOLUME','REVENUE',  'ORANGE', 'ON_NET', 'MONTANT','FREQUENCE']]
st.write(data.head())

# Add input fields for your features and a validation button at the end of the form
input_type = st.sidebar.radio("Select Your Prefered Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    REGULARITY = st.sidebar.slider("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    DATA_VOLUME = st.sidebar.slider("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    REVENUE = st.sidebar.slider("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ORANGE = st.sidebar.slider("ORANGE", data['ORANGE'].min(), data['ORANGE'].max())
    ON_NET = st.sidebar.slider("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    MONTANT = st.sidebar.slider("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE = st.sidebar.slider("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())

else:
    st.sidebar.header('Input Your Information')
    REGULARITY = st.sidebar.number_input("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    DATA_VOLUME = st.sidebar.number_input("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    REVENUE = st.sidebar.number_input("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ORANGE = st.sidebar.slider("ORANGE", data['ORANGE'].min(), data['ORANGE'].max())
    ON_NET = st.sidebar.slider("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    MONTANT = st.sidebar.slider("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE = st.sidebar.slider("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())

st.header('Input Values')

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'REGULARITY':REGULARITY, 'DATA_VOLUME': DATA_VOLUME, 'REVENUE': REVENUE, 'ORANGE':ORANGE, 'ON_NET':ON_NET, 'MONTANT': MONTANT, 'FREQUENCE':FREQUENCE}])


st.write(input_variable)

# Standard Scale the Input Variable.
for i in input_variable.columns:
    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: ##EE9322; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

# Import your ML model into the streamlit application and start making predictions given the provided features values
if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('CHURNERS Predicted')
    st.image('Done.png', width = 50)
    st.success(f'predicted CHURN with provided information is  {predicted}')




