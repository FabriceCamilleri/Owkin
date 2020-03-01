# All functions created during the data preparation + all the imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import six
import SimpleITK as sitk
import radiomics
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm
from scipy import stats
from lifelines import CoxPHFitter
import metrics_t9gbvr2

def prepareDataSet(setType, keepAllEvent=True):
    # Fonction preparing the Training dataset or the Testing data set taking as argument "Training" or "Testing"
    # and keepALlevent boolean in case we want to get rid of the censored data to perform classical regression
    assert (setType == 'Training' or setType == 'Testing'), 'setType argument should be "Training" or "Testing"'
    assert (keepAllEvent == True or keepAllEvent == False), 'keepAllEvent argument should be "True or False'
    # We get the csv files
    clinics=pd.read_csv(setType+"/features/clinical_data.csv")
    radiomics=pd.read_csv(setType+"/features/radiomics.csv", header=1)
    # We rename the column 0
    radiomics.rename(columns = {'Unnamed: 0':'PatientID'}, inplace = True)
    # We get rid of the first row containing only NaN values
    radiomics=radiomics.drop([0])
    # We transform the column PatientID as numeric
    radiomics['PatientID']=pd.to_numeric(radiomics['PatientID'])
    # We create the dataSet
    dataSet=clinics
    dataSet=dataSet.merge(radiomics,on=['PatientID','PatientID'])
    # We set the index as being PatientID
    dataSet=dataSet.set_index('PatientID')
    # We rename some histology values
    dataSet['Histology'][dataSet['Histology']=='Adenocarcinoma']='adenocarcinoma'
    dataSet['Histology'][dataSet['Histology']=='Squamous cell carcinoma']='squamous cell carcinoma'
    #dataSet['Histology'][dataSet['Histology']=='NSCLC NOS (not otherwise specified)']='nos'

    # Standardization of the dataset
    
    #We create binary columns for Histology and Sourcedataset
    dataSet["Adenocarcinoma"] = np.where(dataSet['Histology']=='adenocarcinoma', 1, 0)
    dataSet["Squamous"] = np.where(dataSet['Histology']=='squamous cell carcinoma', 1, 0)
    dataSet["Nos"] = np.where(dataSet['Histology']=='nos', 1, 0)
    dataSet["Nscl"] = np.where(dataSet['Histology']=='NSCLC NOS (not otherwise specified)', 1, 0)
    dataSet["Largecell"] = np.where(dataSet['Histology']=='large cell', 1, 0)
    dataSet["SourceDataset"] = np.where(dataSet['SourceDataset']=='l2', 1, 0)
    # We delete the first column Histology
    dataSet.pop('Histology')
    # We standardize
    dataSet = (dataSet - dataSet.mean())/dataSet.std()
    # We set the missing ages to zero
    dataSet.age[dataSet.age.isnull()]=0  
    # If we want to create a Training set, we add as the last column the known Survival Time
    # They are not known for the Testing set. The only way to get a metric s to submit the output to challengedata.ens.f
    if (setType=='Training'):
        outputs=pd.read_csv("training/output_VSVxRFU.csv")
        dataSet=dataSet.merge(outputs,on=['PatientID','PatientID'])
        dataSet=dataSet.set_index('PatientID')
        if (keepAllEvent==False):
            dataSet=dataSet[dataSet['Event']==1]
            dataSet=dataSet.iloc[:,0:-1]
    return dataSet


# This function will create a datafram with PatientID as index, and with all calculated features with pyradiomics
# For the feature name we keep the naming convention used in the provided data
def calculateRadiomics(setType):
    assert (setType == 'Training' or setType == 'Testing'), 'setType argument should be "Training" or "Testing"'
    dataSet=pd.read_csv(setType+"/features/clinical_data.csv").set_index('PatientID')
    for i in dataSet.index:
        scandata= np.load(setType+'/images/patient_'+'0'*(3-len(str(i)))+str(i)+'.npz')
        # if the mask is not empty, else we don't do anything           
        if ((np.where(scandata['mask'], 1, 0)).max()==1):
            image= sitk.GetImageFromArray(scandata['scan'])
            mask = np.where(scandata['mask'], 1, 0)
            mask = sitk.GetImageFromArray(mask)

            # getting first order features
            firstOrderFeatures=firstorder.RadiomicsFirstOrder(image, mask)
            firstOrderFeatures.enableAllFeatures()
            result = firstOrderFeatures.execute()
            for (key, val) in six.iteritems(result):
                   dataSet.loc[i,'original_firstorder_'+key]=val

            # getting the shape features
            shapeFeatures = shape.RadiomicsShape(image, mask)
            shapeFeatures.enableAllFeatures()
            result = shapeFeatures.execute()
            for (key, val) in six.iteritems(result):
                dataSet.loc[i,'original_shape_'+key]=val

            # getting the Glcm features
            glcmFeatures = glcm.RadiomicsGLCM(image, mask)
            glcmFeatures.enableAllFeatures()
            result = glcmFeatures.execute()
            for (key, val) in six.iteritems(result):
                dataSet.loc[i,'original_glcm_'+key]=val

            # getting the Glrml features
            glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask)
            glrlmFeatures.enableAllFeatures()
            result = glrlmFeatures.execute()
            for (key, val) in six.iteritems(result):
                dataSet.loc[i,'original_glrlm_'+key]=val

            # getting the Glszm features - not used 
            '''
            glszmFeatures = glszm.RadiomicsGLSZM(image, mask)
            glszmFeatures.enableAllFeatures()
            result = glszmFeatures.execute()
            for (key, val) in six.iteritems(result):
                dataSet.loc[i,'original_glszm_'+key]=val
            '''
    # We get rid of the first columns to keep only the calculated data
    dataSet=dataSet.iloc[:,6:]
    
    # We standardize
    dataSet = (dataSet - dataSet.mean())/dataSet.std()
    return dataSet


def getMetric (PredictedOutput, y_true_times):
    predicted_times=PredictedOutput
    predicted_times['Event']=np.nan
    #predicted_times2=predicted_times.append(y_true_times[y_true_times['Event']==0])
    return metrics_t9gbvr2.cindex(y_true_times, predicted_times, tol=1e-8)


def checkData (set):
    for col in set:
        print ('####### Feature: '+ col + ' #######')
        print(set[col].describe())
        if (set[col].dtypes!='object'):
            print ('** Values for which the Z score is greather than 3 : **' )
            print(set[np.abs(stats.zscore(set[col]))>3][col])
            print ('** 5 smalles values: **')
            print(set[col].nsmallest(5))
            print ('** 5 highest values: **')
            print(set[col].nlargest(5)) 
            plt.figure(figsize=(3,3))
            plt.title(col)
            plt.boxplot(set[col][set[col].notnull()])
            plt.show
        else:
            print('** Values and Frequencies: **')
            print(set.groupby(col).size())

        print('** Number of NA: **')
        print(sum(set[col].isnull()))
        print('')
        
        
def mergeDataset (providedSet, calculatedSet):
    providedDataSet=providedSet
    calculatedDataSet=calculatedSet
    for col in calculatedDataSet:
        if col in providedDataSet.columns :
            providedDataSet[col][calculatedDataSet[col].notnull()]=calculatedDataSet[col][calculatedDataSet[col].notnull()]
    return providedDataSet

