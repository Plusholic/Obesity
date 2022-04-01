import pandas as pd
from Data_PreProcessing import Preprocessor, Classifier
import time
import datetime

OS_Method = 'None'
CF_Method = 'SVC'#[LGBM, RFC, 'LGBM',RFC,GBM '', MLP]#,"KNN",LR ''] ', 
day = '0331_Notoversampling_SMOTE_RESULT_'
start = time.time()                       
pl_acc = []
ct = 0
# Preprocessor_ = Preprocessor(top_count = None, ii = None, age = None)

for top_count in [8]:
    pl = []
    pr_list,re_list = [],[]
    fpr_list,tpr_list = [],[]
    auc_list, pr_auc_list = [],[]
    pl_acc.append([])
    for ii in [1,2]: #gender
        for age in [[19, 39], [40, 59], [60, 79]]:
            
            ## Preprocessing
            Preprocessor_ = Preprocessor(top_count, ii, age)
            _, column_feature = Preprocessor_.Load_data()
            Preprocessor_.Age_and_Gender()
            train, test, cnt = Preprocessor_.OverSampler(OS_Method)

            ## Classifier
            Classifier_ = Classifier(top_count, ii, age, OS_Method, train, test, column_feature)
            Classifier_.fit(CF_Method)
            TF, eval, eval2= Classifier_.model_eval()
            csv_list = Classifier_.save_eval(day, cnt)
            # pl_acc[top_count-1].append(eval[0]) # 1부터 끝까지 볼때
            pl_acc[ct].append(eval[0])
            fpr_list.append(eval2[0])
            tpr_list.append(eval2[1])
            re_list.append(eval2[2])
            pr_list.append(eval2[3])
            
            auc_list.append(eval2[4])
            pr_auc_list.append(eval2[5])
                
            pl.append(csv_list)
    ct += 1    
    pl = pd.DataFrame(pl,
                        columns=['gender',
                                '<= age <',
                                "list",
                                'Before Number of 0',
                                'Before Number of 1',
                                'After Number of 0',
                                'After Number of 1',
                                'Train_data_size',
                                'Test_data_size',
                                'TP',
                                'FP',
                                'TN',
                                'FN',
                                'Cross val score',
                                'accuracy score',
                                'recall score',
                                'specificity score',
                                'precision score',
                                'f1 score',
                                'AUROC',
                                'AUPRC'])
    label = [True, False]
    Classifier_.ROCPR_save_sub(fpr_list, tpr_list, re_list, pr_list, label[0])
    
    pl.to_csv('c:/Users/bm990/Desktop/백업/Python_Code/Obesity/2022-01-06/' +
        day + '/Binary_' + CF_Method + '_Result_' + OS_Method +
        '/top ' + str(top_count) +
        ' ' + CF_Method + '_혈액검사 데이터_' + OS_Method + '.csv',
        index=False)

# print(pl_acc)
# print(len(pl_acc))
Classifier_.NFeatureAcc(accuracy=pl_acc, Label=label[1])


sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print(times)

