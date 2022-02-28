from Pipeline import titanic_pipe
from sklearn.metrics import accuracy_score, roc_auc_score
from Train_Test_split import test_train

from Load_Data import load_model_pipeline

titanic_pipe = load_model_pipeline()



X_test,y_test = test_train('test')

class_ = titanic_pipe.predict(X_test)
pred = titanic_pipe.predict_proba(X_test)[:,1]

# determine mse and rmse
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()

# python Test_Predict.py