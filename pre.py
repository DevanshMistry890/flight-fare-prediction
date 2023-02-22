# This file is showing how model process data form raw to training,
# for modelling there is saperate file
# importing Required libraries
import logging
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug(' Pre.py File execution started ')

# loading database with pandas library
df = pd.read_csv("./dataset/Data_clean.csv")
logging.debug(' Database Loaded ')

df.drop(['P1','P2','P3','P4','P5','P6'],axis=1,inplace=True)
df['Arrival_Date'] = df['Arrival_Date'].dropna()

df['Departure'] = df['Date_of_Journey']+' ' + df['Dep_Time']
df['Departure'] = pd.to_datetime(df['Departure'], format='%d-%m-%Y %H:%M')
df.drop(['Date_of_Journey','Dep_Time'],axis=1,inplace=True)

df['Duration'] = pd.to_timedelta(df['Duration']) #format='%Hh %Mm'
df['Duration_Min'] = df['Duration'].dt.total_seconds() / 60
df['Duration_Min'] = df['Duration_Min'].astype(int)
df.drop(['Duration'],axis=1,inplace=True)

df['Stop_No'] = df['Stop_No'].fillna(round(df['Stop_No'].dropna().mean()))
df['Stop_No'] = df['Stop_No'].astype(int)

df['D_Month'] = df['Departure'].dt.month
df['D_Day'] = df['Departure'].dt.day
df['D_Hour'] = df['Departure'].dt.hour
df['D_Minutes'] = df['Departure'].dt.minute
df.drop(['Departure'],axis=1,inplace=True)

df['Arrival_TM'] = pd.to_datetime(df['Arrival_TM'], format='%H:%M')
df['A_Hour'] = df['Arrival_TM'].dt.hour
df['A_Minutes'] = df['Arrival_TM'].dt.minute
df.drop(['Arrival_TM'],axis=1,inplace=True)

#Arrival_Date
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%b')
df['A_Month'] = df['Arrival_Date'].dt.month
df['A_Day'] = df['Arrival_Date'].dt.day
df.drop(['Arrival_Date'],axis=1,inplace=True)

df['A_Month'] = df['A_Month'].fillna(0)
df['A_Day'] = df['A_Day'].fillna(0)
df['A_Month'] = df['A_Month'].astype(int)
df['A_Day'] = df['A_Day'].astype(int)

#df.loc[df['Additional_Info'] == 'No Info'] = 'No info'

category_col = ['Airline','Source','Destination','Additional_Info']


labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
	print(df[col].unique())
	df[col] = labelEncoder.fit_transform(df[col])
	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping
logging.debug(mapping_dict)
logging.debug('Database Pre-processing is Finished')
df.drop(index=0,axis=1,inplace=True)
df.to_csv('model_pre.csv')
logging.debug('db for modeling is created.')