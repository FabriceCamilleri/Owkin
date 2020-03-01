# Owkin Data challenge

https://challengedata.ens.fr/participants/challenges/33/

## Preparation of the data
The data provided for the data challenge are:
- clinicals data
- radiomics
- scans

After analysing them, a new dataset has been prepared as follow:
- Some Histology values have been updated to match the correct category (case issue)
- Histology values haven been used to create specific columns
- Histology fetaure has then been deleted
- Sourcedataset has been recoded: 0 for l1 and 1 for l2
- All the features have been standardized, except for SurvivalTime and Event
- Radiomics data were calculated automatically using pyradiomics and standardized
- when a calculated radiomics feature matches with a provided radiomic feature, the calculated ones have been used to replace the provided
ones, when available, as it was not possible to do the calculation for empty mask. So when not available, the provided values has been used
- Age missing values have been replaces by 0


## Feature selections
Feature selection has been done in R using the glmnet library. 
A cross validation has been performed, leaving R trying several lambdas
A final model has been trained with the min of lambda + 1 standard error.

## Cox model
A cox model has been trained using the features selected in R, with an addiotional ridge parameter has been used to avoid overfitting


## Conclusion and next steps
Higher score: 07255
As next steps, we could:
- impute age missing values differently, for example by trying to cluser the obervations and compute a mean per cluster
- try to calculate all radiomics features 
- ask expert about the missing masks for some scans


