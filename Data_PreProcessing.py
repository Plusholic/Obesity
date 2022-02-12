import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.preprocessing import label_binarize
from collections import Counter
plt.rcParams["font.family"] = "Palatino Linotype"



list999 = []
list88 = []
elselist = []
column_feature = []
accuracy = [] 
count0 = [] 
count1 = []
print_list = []
# BMI_grade = []


class Preprocessor:
    def __init__(self, top_count, ii, age):
        
        self.top_count = top_count
        self.ii = ii
        self.age = age
        
        self.sex = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None
        self.BMI_grade = []
    
    def Load_data(self):
        # self.data = self.data
        data1 = pd.read_csv("c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/DATA/hn2016_all.csv",encoding='utf-8', low_memory=False)
        data2 = pd.read_csv("c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/DATA/hn2017_all.csv",encoding='utf-8', low_memory=False)
        data3 = pd.read_csv("c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/DATA/hn2018_all.csv",encoding='utf-8', low_memory=False)
        data4 = pd.read_csv("c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/DATA/hn2019_all.csv",encoding='utf-8', low_memory=False)
        
        self.data = pd.concat([data1, data2, data3, data4], ignore_index=True) # 18년 19년 자료 합쳐주는 부분.
        
        # Feature_Selection = pd.read_csv('RFC_Feature_Selection/RFC_feature_selection_Binary_OVERSAMPLING_No_gender_No_age_ADASYN.csv', index_col = 0)
        Feature_Selection = pd.read_csv('c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/RFC_Feature_Selection/RFC_feature_selection_Binary_OverSampling.csv', index_col = 0)
        filtering = Feature_Selection[(Feature_Selection['gender'] == self.ii) & (Feature_Selection['age'] == str(self.age))]
        column_feature = ['HE_BMI'] + list(filtering.index[0:self.top_count])
        self.column_feature = column_feature
        
        return self.data, self.column_feature
        
        
    def Age_and_Gender(self):#, top_count=None, ii=None, age = [19, 39]):
        
        # 성별 지정, 나이 지정해서 for loop로 돌려야함.
        gender = self.data['sex']
        
        data_copy = self.data[(self.data['age']>=self.age[0]) & (self.data['age']<self.age[1])].copy()
    
        sex = [self.ii]
        data_copy = data_copy.loc[gender.isin(sex)]

        # # Feature_Selection = pd.read_csv('RFC_Feature_Selection/RFC_feature_selection_Binary_OVERSAMPLING_No_gender_No_age_ADASYN.csv', index_col = 0)
        # Feature_Selection = pd.read_csv('c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/RFC_Feature_Selection/RFC_feature_selection_Binary_OverSampling.csv', index_col = 0)
        # filtering = Feature_Selection[(Feature_Selection['gender'] == self.ii) & (Feature_Selection['age'] == str(self.age))]
        # column_feature = ['HE_BMI'] + list(filtering.index[0:self.top_count])
        # self.column_feature = column_feature
        
        
        data_select = data_copy[self.column_feature].copy()
        for i in range(len(self.column_feature)):
            # self.BMI_grade.append([])
            ## 숫자로 바꿔주는 코드임.
            data_select[self.column_feature[i]] = pd.to_numeric(data_select[self.column_feature[i]], errors='coerce').astype(float).round(2)          #print(len(df)) #16000개
            

        df = data_select[self.column_feature]
        df = df.dropna(how = 'any')
        df = df.sort_values(by = 'HE_BMI')
        for i in range(len(self.column_feature)):
            ### 8,9제거
            if self.column_feature[i] in list999:
                df.drop(df[(df[self.column_feature[i]] == 888) | (df[self.column_feature[i]] == 999)].index, inplace = True)
            elif self.column_feature[i] in list88:
                df.drop(df[(df[self.column_feature[i]] == 88) | (df[self.column_feature[i]] == 99)].index, inplace = True)
            else:
                df.drop(df[(df[self.column_feature[i]] == 8) | (df[self.column_feature[i]] == 9)].index, inplace = True)
        # print(df['HE_BMI'])
        BMI_tmp = df['HE_BMI'].astype(float)
        # return print(len(df))
        for k in range(len(df)):
            # print(BMI_tmp.iloc[k])
            if BMI_tmp.iloc[k] < 23:
                self.BMI_grade.append(0)
            elif 23 <= BMI_tmp.iloc[k]:
                self.BMI_grade.append(1)
        tree_data = df.drop(['HE_BMI'],axis = 1)
        
        # iteration_test.append([iter,ct])
        if tree_data.empty == False:
            # data normalizaion
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(tree_data)
            tree_data = pd.DataFrame(x_scaled,columns=tree_data.columns)
            tree_data['BMI_grade'] = self.BMI_grade#[iter]
            from sklearn.model_selection import train_test_split
            X = tree_data.iloc[:,:-1]
            y = tree_data.iloc[:,-1:]
            # y = y.squeeze()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        return tree_data
    
    def OverSampler(self, Method=None):
        cnt_list = self.y_train['BMI_grade'].squeeze()
        cnt_list = cnt_list.tolist()
        cnt1 = cnt_list.count(1)
        cnt0 = cnt_list.count(0)
        print(f'before augmentation : 0 : {cnt0}, 1 : {cnt1}')
        self.Method = Method
        from collections import Counter
        if Method == None:
            from imblearn.over_sampling import RandomOverSampler
            OS = RandomOverSampler(random_state=42)
            self.X_train, self.y_train = OS.fit_resample(self.X_train, self.y_train)
            
        elif Method == "SMOTE":
            from imblearn.over_sampling import SMOTE
            OS = SMOTE(random_state=42)#, ratio = 1.0)
            self.X_train, self.y_train = OS.fit_resample(self.X_train, self.y_train)
            
        elif Method == "ADASYN":
            from imblearn.over_sampling import ADASYN
            cnt_1, cnt_0 = 0, 0
            for i2 in self.y_train.index:
                if self.y_train['BMI_grade'][i2] == 0:
                    cnt_0 += 1
                elif self.y_train['BMI_grade'][i2] == 1:
                    cnt_1 += 1
            if(Counter(self.y_train['BMI_grade'])[0]/Counter(self.y_train['BMI_grade'])[1] > 1.1) | (Counter(self.y_train['BMI_grade'])[1]/Counter(self.y_train['BMI_grade'])[0] > 1.1):
                OS = ADASYN(sampling_strategy='minority', random_state=42)
                self.X_train, self.y_train = OS.fit_resample(self.X_train, self.y_train)
        
        cnt_dict = dict(Counter(self.y_train['BMI_grade'].ravel()))
        cnt0 = cnt_dict[0]
        cnt1 = cnt_dict[1]
        print(f'after augmentation : 0 : {cnt0}, 1 : {cnt1}')
        
        return (self.X_train, self.y_train), (self.X_test, self.y_test), (cnt0, cnt1)
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

class Classifier:
    def __init__(self, top_count, ii, age, Method, train, test, column_feature):

        self.column_feature = column_feature
        self.Method = Method
        self.top_count = top_count
        self.ii = ii
        self.age = age
        self.X_train = train[0]
        self.y_train = train[1]
        self.X_test = test[0]
        self.y_test = test[1]
        
        self.clf = None
        self.y_pred, self.r_score, self.p_score, self.f_score, self.accuracy = None,None,None,None,None
                
    def perf_measure(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return(TP, FP, TN, FN)

    def fit(self, model = None):#, X_train, y_train):
        self.model = model
        
        if model == 'SGD':    
            from sklearn.linear_model import SGDClassifier
            self.clf = SGDClassifier(max_iter=1000, random_state = 42)
            self.clf.fit(self.X_train, self.y_train.values.ravel())
            
            
        elif model == 'GBM':
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import GridSearchCV
            
            clf = GradientBoostingClassifier(random_state=42)
            parameters = {'n_estimators':[3,10,20,30,40,50,70,100],
                        'max_depth':[2,3,5,10,30,50,70,100]}
            grid_dtree = GridSearchCV(clf,param_grid=parameters,cv=3,refit=True)
            grid_dtree.fit(self.X_train, self.y_train.values.ravel())

            print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
            print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))

            best_param = list(grid_dtree.best_params_.values())

            clf = GradientBoostingClassifier(n_estimators= best_param[1] ,max_depth= best_param[0], random_state=42)
            clf.fit(self.X_train, self.y_train.values.ravel())
            
            
        elif model == 'RFC':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GridSearchCV
            
            clf = RandomForestClassifier(random_state=42)
            parameters = {'n_estimators':[3,10,20,30,40,50,70,100],
                        'max_depth':[2,3,5,10,30,50,70,100]}
            grid_dtree = GridSearchCV(clf,param_grid=parameters,cv=3,refit=True)
            grid_dtree.fit(self.X_train, self.y_train.values.ravel())

            print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
            print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))
            best_param = list(grid_dtree.best_params_.values())
            clf = RandomForestClassifier(n_estimators= best_param[1] ,max_depth= best_param[0], random_state=42)
            clf.fit(self.X_train, self.y_train.values.ravel())
            
  
    def model_eval(self):
        
        self.y_pred = self.clf.predict(self.X_test) 
        self.r_score = recall_score(self.y_test, self.y_pred)
        self.p_score = precision_score(self.y_test, self.y_pred)
        self.f_score = f1_score(self.y_test, self.y_pred)
        self.accuracy = self.clf.score(self.X_test, self.y_test)
                
        self.precision, self.recall, _ = precision_recall_curve(self.y_test, self.clf.decision_function(self.X_test))
        self.fpr, self.tpr, _ = roc_curve(self.y_test,self.clf.decision_function(self.X_test))
        
        self.TP = self.perf_measure(np.array(self.y_test), self.y_pred)[0]
        self.FP = self.perf_measure(np.array(self.y_test), self.y_pred)[1]
        self.TN = self.perf_measure(np.array(self.y_test), self.y_pred)[2]
        self.FN = self.perf_measure(np.array(self.y_test), self.y_pred)[3]
        
        return (self.TP, self.FP, self.TN, self.FN), (self.accuracy, self.r_score, self.p_score, self.f_score)
        
    def save_eval(self, day, cnt):
        cnt0 = cnt[0]
        cnt1 = cnt[1]
        
        sup = "_" + str(self.Method)
        PATH = "2022-01-06/" + day + "/Binary_" + str(self.model) + "_Result" + sup
        PATH2 = PATH + "/eps"
        import os
        os.makedirs(day,exist_ok=True)
        os.makedirs(PATH,exist_ok=True)
        os.makedirs(PATH2,exist_ok=True)

        print_list = []

        # print_list.append([])
        print_list.append(str(self.ii))
        print_list.append(str(self.age))
        print_list.append(str(self.column_feature))
        print_list.append(str(cnt0))
        print_list.append(str(cnt1))
        
        print_list.append(str(len(self.y_train))) # train_data_size
        print_list.append(str(len(self.y_test))) # test_data_size
        
        print_list.append(self.TP)
        print_list.append(self.FP)
        print_list.append(self.TN)
        print_list.append(self.FN)
        print_list.append(np.round(self.accuracy,3))
        print_list.append(np.round(self.r_score,3))
        print_list.append(np.round(self.p_score,3))
        print_list.append(np.round(self.f_score,3))
        
        ## ROC CURVE
        plt.figure()
        plt.plot(self.fpr, self.tpr,label = "roc curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc = 4)
        plt.fill_between(self.fpr, self.tpr, alpha=0.5)
        clf_auc = roc_auc_score(self.y_test, self.clf.decision_function(self.X_test))
        plt.text(0.65,0.2,"AUC score: {:.3f}".format(clf_auc))
        plt.title('top ' + str(self.top_count) + ' ' + str(self.ii) + ' ' + str(self.age) + ' SGD')
        
        plt.savefig(PATH +'/top ' + str(self.top_count)
                    + ' ' + str(self.age)
                    + ' ' + str(self.ii) + ' SGD' + sup + '.png')
        plt.savefig(PATH2 +'/top ' + str(self.top_count)
                    + ' ' + str(self.age)
                    + ' ' + str(self.ii) + ' SGD' + sup + '.pdf')
        plt.close()
        
        ## Confusion Matrix
        y_train_pred = cross_val_predict(self.clf, self.X_train, self.y_train.values.ravel())
        conf_mx = confusion_matrix(self.y_train.values.ravel(), y_train_pred)
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.savefig(PATH +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) + ' SGD_confusion-matrix' + sup + '.png')
        plt.savefig(PATH2 +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) + ' SGD_confusion-matrix' + sup + '.pdf')
        
        # Precision - Recall Curve
        plt.figure()
        plt.plot(self.recall, self.precision, alpha = 0.5)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.savefig(PATH +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) + ' SGD_Precision-recall Curve' + sup + '.png')
        plt.savefig(PATH2 +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) + ' SGD_Precision-recall Curve' + sup + '.pdf')        
            
        # print(print_list)
        # pl = pd.DataFrame(print_list,
        #                     columns=['gender',
        #                             '<= age <',
        #                             # "group",
        #                             "list",
        #                             'Number Of 0',
        #                             'Number of 1',
        #                             'Train_data_size',
        #                             'Test_data_size',
        #                             'TP',
        #                             'FP',
        #                             'TN',
        #                             'FN',
        #                             'accuracy score',
        #                             'recall score',
        #                             'precision score',
        #                             'f1 score'])
        return print_list

    # pl.to_csv(PATH + '/top ' + str(top_count) + ' SGD_혈액검사 데이터' + sup + '.csv', index=False)
    
    # def FunctionName(args):
    #     if sex == [1]:  
    #         if age == 0:
    #             acc1939_1_list.append(np.round(self.accuracy,3))
    #             f1939_1_list.append(np.round(self.f_score,3))
    #         elif age == 1:
    #             acc3959_1_list.append(np.round(self.accuracy,3))
    #             f3959_1_list.append(np.round(self.f_score,3))
    #         elif age == 2:
    #             acc5979_1_list.append(np.round(self.accuracy,3))
    #             f5979_1_list.append(np.round(self.f_score,3))
    #     elif sex == [2]:
    #         if age == 0:
    #             acc1939_2_list.append(np.round(self.accuracy,3))
    #             f1939_2_list.append(np.round(self.f_score,3))
    #         elif age == 1:
    #             acc3959_2_list.append(np.round(self.accuracy,3))
    #             f3959_2_list.append(np.round(self.f_score,3))
    #         elif age == 2:
    #             acc5979_2_list.append(np.round(self.accuracy,3))
    #             f5979_2_list.append(np.round(self.f_score,3))  
        