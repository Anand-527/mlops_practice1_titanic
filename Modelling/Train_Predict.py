
from Pipeline import titanic_pipe
from sklearn.metrics import accuracy_score, roc_auc_score
from Train_Test_split import test_train
from Load_Data import save_model_pipeline

titanic_pipe = titanic_pipe()


X_train,y_train = test_train('train')

titanic_pipe.fit(X_train,y_train)

save_model_pipeline(titanic_pipe)


# make predictions for train set
class_ = titanic_pipe.predict(X_train)
pred = titanic_pipe.predict_proba(X_train)[:,1]

# determine mse and rmse
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# python Train_Predict.py