Model training
     ↓
model.fit()
     ↓
Model learns coefficients
     ↓
model.predict()
     ↓
y_pred
     ↓
Compare y_test vs y_pred
     ↓
MSE / MAE / R²

the ,se MAE and r 2 are evelaution matrices ,

1. MSE : take the squared erros and averages them 
MSE=n1​i=1∑n​(yi​−y^​i​)2

2. MAE  mean absolute erro , it takes the absoulte value not the sequre value 
MAE=n1​i=1∑n​∣yi​−y^​i​∣
tell average size of orection error 

3. R2 :  it will jsut precit the mean  (COEFFICINET OF DETERMINATION )
1.0 → excellent/perfect fit
0 → roughly no improvement over predicting the mean
< 0 → worse than that simple mean baseline
________________________________________________________
main conepcts residuls , loss function , evalution metirces , 
all uses differcne between actual and predcit output , but somehwt differnet formula , base idea is the same 

Concept	            What it does with the difference
Residual	        Stores/describes the individual error
Loss function	    Converts error into a value used to guide learning
Evaluation metric	Summarizes errors to judge model performance
_______________________________________________________
R² — Coefficient of Determination : HOW MUCH variation in the target variable is epxlained by regression model 
R2=1−  (∑(yi​−yˉ​)2  / ∑(yi​−y^​i​)2​)
_________________________________________________________
assumptions of linear regression : 

1. linearity : the relationship between x and y shld be linear 
2. indepnedence : erros shld be idnepnecent of another