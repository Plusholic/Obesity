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
    def __init__(self, top_count=None, ii=None, age=None):
        
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
            
            y = label_binarize(y, classes=[0, 1])
            # n_classes = y.shape[1]
                
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        return tree_data
    
    def OverSampler(self, Method=None):
        
        cnt_array = np.array(self.y_train)
        cnt_array = cnt_array.reshape(1,-1)
        cnt_array = cnt_array.tolist()

        cnt0 = cnt_array[0].count(0)
        cnt1 = cnt_array[0].count(1)

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
            for i2 in range(len(self.y_train)):
                if self.y_train[i2] == 0:
                    cnt_0 += 1
                elif self.y_train[i2] == 1:
                    cnt_1 += 1
            if(cnt_0/cnt_1 > 1.1) | (cnt_1/cnt_0 > 1.1):
                OS = ADASYN(sampling_strategy='minority', random_state=42)
                self.X_train, self.y_train = OS.fit_resample(self.X_train, self.y_train)
        
        train_data_size = len(self.y_train)
        test_data_size = len(self.y_test)
            
        cnt_dict = dict(Counter(self.y_train))
        
        cnt0 = cnt_dict[0]
        cnt1 = cnt_dict[1]
        print(f'after augmentation : 0 : {cnt0}, 1 : {cnt1}')
        print(' ')
        
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
            self.clf.fit(self.X_train, self.y_train)
            
            
        elif model == 'GBM':
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import GridSearchCV
            
            self.clf = GradientBoostingClassifier(random_state=42)
            parameters = {'n_estimators':[3,10,20,30,40,50,70,100],
                        'max_depth':[2,3,5,10,30,50,70,100]}
            grid_dtree = GridSearchCV(self.clf,param_grid=parameters,cv=3,refit=True)
            grid_dtree.fit(self.X_train, self.y_train)

            print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
            print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))

            best_param = list(grid_dtree.best_params_.values())

            self.clf = GradientBoostingClassifier(n_estimators= best_param[1] ,max_depth= best_param[0], random_state=42)
            self.clf.fit(self.X_train, self.y_train)
            
            
        elif model == 'RFC':
            # from sklearn.ensemble import RandomForestClassifier
            from sklearn import ensemble
            from sklearn.model_selection import GridSearchCV
            
            self.clf = ensemble.RandomForestClassifier(random_state=42)
            parameters = {'n_estimators':[3,10,20,30,40,50,70,100],
                        'max_depth':[2,3,5,10,30,50,70,100]}
            grid_dtree = GridSearchCV(self.clf,param_grid=parameters,cv=3,refit=True)
            grid_dtree.fit(self.X_train, self.y_train)

            print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
            print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))
            best_param = list(grid_dtree.best_params_.values())
            self.clf = ensemble.RandomForestClassifier(n_estimators= best_param[1] ,max_depth= best_param[0], random_state=42)
            self.clf.fit(self.X_train, self.y_train)
            
        elif model == 'MLP':
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import GridSearchCV
            
            self.clf = MLPClassifier(random_state=42, max_iter=10000, activation='relu',hidden_layer_sizes=[100, 100])
            # clf1.fit(X_train,y_train)
            
            parameters = {'alpha':[0.0001, 0.001, 0.01, 0.1]}
            grid_dtree1 = GridSearchCV(self.clf,param_grid=parameters,cv=3,refit=True)
            grid_dtree1.fit(self.X_train, self.y_train)
            best_param1 = list(grid_dtree1.best_params_.values())
            
            self.clf = MLPClassifier(random_state=42, max_iter=1000, activation='relu',hidden_layer_sizes=[100, 100],alpha= best_param1[0])
            self.clf.fit(self.X_train, self.y_train)    
            
  
    def model_eval(self):
        
        self.y_pred = self.clf.predict(self.X_test)
        self.r_score = recall_score(self.y_test, self.y_pred)
        self.p_score = precision_score(self.y_test, self.y_pred)
        self.f_score = f1_score(self.y_test, self.y_pred)
        self.accuracy = self.clf.score(self.X_test, self.y_test)
        
        if self.model in ['RFC', 'MLP', 'GBM']:
            self.precision, self.recall, _ = precision_recall_curve(self.y_test, self.clf.predict_proba(self.X_test)[:, 1])
            self.fpr, self.tpr, _ = roc_curve(self.y_test,self.clf.predict_proba(self.X_test)[:, 1])
        
        elif self.model == 'SGD':
            
            self.precision, self.recall, _ = precision_recall_curve(self.y_test, self.clf.decision_function(self.X_test))
            self.fpr, self.tpr, _ = roc_curve(self.y_test,self.clf.decision_function(self.X_test))
        
        self.TP = self.perf_measure(np.array(self.y_test), self.y_pred)[0]
        self.FP = self.perf_measure(np.array(self.y_test), self.y_pred)[1]
        self.TN = self.perf_measure(np.array(self.y_test), self.y_pred)[2]
        self.FN = self.perf_measure(np.array(self.y_test), self.y_pred)[3]
        
        return (self.TP, self.FP, self.TN, self.FN), (self.accuracy, self.r_score, self.p_score, self.f_score), (self.fpr, self.tpr, self.recall, self.precision)
        
    def save_eval(self, day, cnt):
        cnt0 = cnt[0]
        cnt1 = cnt[1]
        
        self.sup = "_" + str(self.Method)
        self.PATH = "2022-01-06/" + day + "/Binary_" + str(self.model) + "_Result" + self.sup
        self.PATH2 = self.PATH + "/eps"
        import os
        os.makedirs(day,exist_ok=True)
        os.makedirs(self.PATH,exist_ok=True)
        os.makedirs(self.PATH2,exist_ok=True)

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
        specificity = self.TN / (self.TN+self.FP)
        print(self.TP / (self.TP + self.FN))
        print(self.r_score)
        print_list.append(np.round(specificity,3))
        print_list.append(np.round(self.p_score,3))
        print_list.append(np.round(self.f_score,3))
        
        ## ROC CURVE
        plt.figure()
        plt.plot(self.fpr, self.tpr,label = "roc curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc = 4)
        plt.fill_between(self.fpr, self.tpr, alpha=0.5)
        
        if self.model in ['RFC', 'MLP', 'GBM']:
            clf_auc = roc_auc_score(self.y_test, self.clf.predict_proba(self.X_test)[:, 1])
        elif self.model == 'SGD':
            clf_auc = roc_auc_score(self.y_test, self.clf.decision_function(self.X_test))
        
        plt.text(0.65,0.2,"AUC score: {:.3f}".format(clf_auc))
        plt.title('top ' + str(self.top_count) + ' ' + str(self.ii) + ' ' + str(self.age) + ' ' + self.model)
        
        plt.savefig(self.PATH +'/top ' + str(self.top_count)
                    + ' ' + str(self.age)
                    + ' ' + str(self.ii) +  ' ' + self.model + self.sup + '.png')
        plt.savefig(self.PATH2 +'/top ' + str(self.top_count)
                    + ' ' + str(self.age)
                    + ' ' + str(self.ii) +  ' ' + self.model + self.sup + '.pdf')
        plt.close()
        
        # ## Confusion Matrix
        y_train_pred = cross_val_predict(self.clf, self.X_train, self.y_train)
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.savefig(self.PATH +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) +  ' ' + self.model + '_confusion-matrix' + self.sup + '.png')
        plt.savefig(self.PATH2 +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) +  ' ' + self.model + '_confusion-matrix' + self.sup + '.pdf')
        
        # Precision - Recall Curve
        plt.figure()
        plt.plot(self.recall, self.precision, alpha = 0.5)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.savefig(self.PATH +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) +  ' ' + self.model + '_Precision-recall Curve' + self.sup + '.png')
        plt.savefig(self.PATH2 +"/Top " + str(self.top_count)
                    + " " + str(self.age)
                    + " " + str(self.ii) +  ' ' + self.model + '_Precision-recall Curve' + self.sup + '.pdf')        
            
        return print_list
    
    def NFeatureAcc(self, accuracy=accuracy):
        acc1939_1_list = []
        acc3959_1_list = []
        acc5979_1_list = []
        acc1939_2_list = []
        acc3959_2_list = []
        acc5979_2_list = []
        
        for i in range(len(accuracy)):
            acc1939_1_list.append(accuracy[i][0])
            acc3959_1_list.append(accuracy[i][1])
            acc5979_1_list.append(accuracy[i][2])
            acc1939_2_list.append(accuracy[i][3])
            acc3959_2_list.append(accuracy[i][4])
            acc5979_2_list.append(accuracy[i][5])
        
        plt.rcParams["font.family"] = "Palatino Linotype"
        plt.figure()
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        plt.plot(acc1939_1_list, ':D', label = 'man1939', linewidth=3, color='firebrick')
        plt.plot(acc3959_1_list, ':D', label = 'man3959', linewidth=3, color='tomato')
        plt.plot(acc5979_1_list, ':D', label = 'man5979', linewidth=3, color='saddlebrown')
        plt.plot(acc1939_2_list, '--o', label = 'women1939', linewidth=3, color='royalblue')
        plt.plot(acc3959_2_list, '--o', label = 'women3959', linewidth=3, color='dodgerblue')
        plt.plot(acc5979_2_list, '--o', label = 'women5979', linewidth=3, color='slateblue')
        plt.xticks([0,1,2,3,4,5,6,7,8,9])
        plt.legend(['Male 19-39','Male 39-59','Male 59-79','Female 19-39','Female 39-59','Female 59-79'],
                fontsize=15, loc="lower right", ncol=2)
        plt.yticks(fontsize=20)
        plt.ylabel('Accuracy', fontsize=25)
        plt.xlabel('Top Feature', fontsize=25)
        ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'], fontsize=20)
        plt.tight_layout()
        plt.savefig(self.PATH +"/ " + self.model + '_Accuracy per feature number' + '.png')
        plt.savefig(self.PATH2 +"/ " + self.model + '_Accuracy per feature number' + '.pdf')
        


    def ROCPR_save_sub(self, fpr_list, tpr_list, re_list, pr_list):
        from matplotlib import cm
        roc_color = cm.OrRd(np.linspace(0,1,10))
        pr_color = cm.PuBu(np.linspace(0,1,10))
        plt.figure()
        fig, big_axes = plt.subplots(2,1, figsize=(16,16))
        for row, big_ax in enumerate(big_axes, start=1):
        #     big_ax.set_title("Subplot row %s \n" % row, fontsize=16)

            # Turn off axis lines and ticks of the big subplot 
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            big_ax._frameon = False

        fig.add_subplot(2,2,1)
        plt.plot(fpr_list[0],tpr_list[0], linewidth=2, color=roc_color[4])
        plt.plot(fpr_list[1],tpr_list[1], linewidth=2, color=roc_color[6])
        plt.plot(fpr_list[2],tpr_list[2], linewidth=2, color=roc_color[9])

        plt.fill_between(fpr_list[0],tpr_list[0], alpha=0.3, color=roc_color[4])
        plt.fill_between(fpr_list[1],tpr_list[1], alpha=0.3, color=roc_color[6])
        plt.fill_between(fpr_list[2],tpr_list[2], alpha=0.3, color=roc_color[9])
        plt.legend(['Male 19-39','Male 39-59','Male 59-79'],
                fontsize=15, loc="lower right")
        plt.xlim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 1])


        fig.add_subplot(2,2,2)
        plt.plot(fpr_list[3],tpr_list[3], linewidth=2, color=roc_color[4])
        plt.plot(fpr_list[4],tpr_list[4], linewidth=2, color=roc_color[6])
        plt.plot(fpr_list[5],tpr_list[5], linewidth=2, color=roc_color[9])
        plt.fill_between(fpr_list[3],tpr_list[3], alpha=0.3, color=roc_color[4])
        plt.fill_between(fpr_list[4],tpr_list[4], alpha=0.3, color=roc_color[6])
        plt.fill_between(fpr_list[5],tpr_list[5], alpha=0.3, color=roc_color[9])
        plt.legend(['Female 19-39','Female 39-59','Female 59-79'],
                fontsize=15, loc="lower right")
        plt.xlim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 1])


        fig.add_subplot(2,2,3)
        plt.plot(re_list[0], pr_list[0], linewidth=2, color=pr_color[4])
        plt.plot(re_list[1], pr_list[1], linewidth=2, color=pr_color[6])
        plt.plot(re_list[2], pr_list[2], linewidth=2, color=pr_color[9])
        plt.fill_between(re_list[0], pr_list[0], alpha=0.3, color=pr_color[4])
        plt.fill_between(re_list[1], pr_list[1], alpha=0.3, color=pr_color[6])
        plt.fill_between(re_list[2], pr_list[2], alpha=0.3, color=pr_color[9])
        plt.xlim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 1])
        plt.legend(['Male 19-39','Male 39-59','Male 59-79'],
                fontsize=15, loc="lower right")


        fig.add_subplot(2,2,4)
        plt.plot(re_list[3], pr_list[3], linewidth=2, color=pr_color[4])
        plt.plot(re_list[4], pr_list[4], linewidth=2, color=pr_color[6])
        plt.plot(re_list[5], pr_list[5], linewidth=2, color=pr_color[9])
        plt.fill_between(re_list[3], pr_list[3], alpha=0.3, color=pr_color[4])
        plt.fill_between(re_list[4], pr_list[4], alpha=0.3, color=pr_color[6])
        plt.fill_between(re_list[5], pr_list[5], alpha=0.3, color=pr_color[9])
        plt.legend(['Female 19-39','Female 39-59','Female 59-79'],
                fontsize=15, loc="lower right")
        plt.xlim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 1])

        big_axes[0].text(0.43, 1.05, self.model + ' Roc Curve', fontsize=25)
        big_axes[1].text(0.37, 1.05, self.model + ' Precision-Recall Curve', fontsize=25)

        plt.tight_layout()
        plt.savefig(self.PATH + "/Top " + str(self.top_count) + " " +
                    self.model + '_ROC-PR Curve' + '.png')
        plt.savefig(self.PATH2 +"/Top " + str(self.top_count) + " " +
                    self.model + '_ROC-PR Curve' + '.pdf')