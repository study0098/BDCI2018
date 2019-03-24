
# coding: utf-8

# In[1]:


import pandas as pd
import xgboost as xgb
import time
import numpy as np
from sklearn import preprocessing
import category_encoders as ce


# In[2]:


entbase_df = pd.read_csv('./data/1entbase.csv')
alter_df = pd.read_csv('./data/2alter.csv')
branch_df = pd.read_csv('./data/3branch.csv')
invest_df = pd.read_csv('./data/4invest.csv')
right_df = pd.read_csv('./data/5right.csv')
project_df = pd.read_csv('./data/6project.csv')
lawsuit_df = pd.read_csv('./data/7lawsuit.csv')
breakfaith_df = pd.read_csv('./data/8breakfaith.csv')
recruit_df = pd.read_csv('./data/9recruit.csv')
qualification_df = pd.read_csv('./data/10qualification.csv')


# In[3]:


def translate_year(date):
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2010)*12 + month
	# 也可以用timedate类型来进行计算，不过好像比较慢？

def get_interaction_feature(df, feature_A, feature_B):
    feature_A_list = sorted(df[feature_A].unique())
    feature_B_list = sorted(df[feature_B].unique())
    count = 0
    mydict = {}
    for i in feature_A_list:
        mydict[int(i)] = {}
		# 字典本应用key进行索引，但是这里用数字进行索引，
		# 本质：创建一个数字类型的key，然后设置其value为{}
        for j in feature_B_list:
            mydict[int(i)][int(j)] = count
            count += 1
	
	# 先构造了一个字典，然后根据df的每一行的情况，去索引里面找一个值
    return df.apply(lambda x: mydict[int(x[feature_A])][int(x[feature_B])], axis=1)
	# axis=1，是对df的每一行x进行处理，x[feature_A]表示该行在该列索引上的取值
	# 最后结果应该是series


# In[4]:


import math

def get_entbase_feature():
    entbase = entbase_df.copy()
    entbase.MPNUM.fillna(0, inplace=True)
    entbase.INUM.fillna(0, inplace=True)
    entbase.FINZB.fillna(0, inplace=True)
    entbase.FSTINUM.fillna(0, inplace=True)
    entbase.TZINUM.fillna(0, inplace=True)
    entbase.ENUM.fillna(0, inplace=True)
    entbase.HY.fillna(51, inplace=True)
    entbase.ZCZB.fillna(7, inplace=True)
    
#     entbase["PROV"] = preprocessing.LabelEncoder().fit_transform(entbase.PROV.values)
    entbase["RGYEAR"] = 2015 - entbase["RGYEAR"]
    entbase["INDEX_SUM"] = entbase["MPNUM"] + entbase["INUM"] + entbase["MPNUM"] + entbase["TZINUM"] + entbase["ENUM"]
    entbase["ZCZB_FINZB_RATE"] = entbase["ZCZB"] / entbase["FINZB"]
    entbase["ZCZB_FINZB_RATE"] = entbase["FINZB"] / entbase["ZCZB"]
    
    entbase["DIVIDE_ZCZB_RGYEAR"] = entbase["ZCZB"] / entbase["RGYEAR"]
    entbase["DIVIDE_RGYEAR_ZCZB"] = entbase["RGYEAR"] / entbase["ZCZB"]
    entbase["MUL_HY_PROV"] = entbase["HY"] * entbase["PROV"]
    entbase["MUL_HY_ETYPE"] = entbase["HY"] * entbase["ETYPE"]
    entbase["MUL_HY_RGYEAR"] = entbase["HY"] * entbase["RGYEAR"]
    
    entbase["RGYEAR_ZCZB"] = get_interaction_feature(entbase, "RGYEAR", "ZCZB")
    entbase["RGYEAR_FINZB"] = get_interaction_feature(entbase, "RGYEAR", "FINZB")
    entbase["RGYEAR_INUM"] = get_interaction_feature(entbase, "RGYEAR", "INUM")
    entbase["RGYEAR_ENUM"] = get_interaction_feature(entbase, "RGYEAR", "ENUM")
    entbase["RGYEAR_FSTINUM"] = get_interaction_feature(entbase, "RGYEAR", "FSTINUM")
    entbase["RGYEAR_TZINUM"] = get_interaction_feature(entbase, "RGYEAR", "TZINUM")
    entbase["RGYEAR_MPNUM"] = get_interaction_feature(entbase, "RGYEAR", "MPNUM")
    
    #new 12.7
    entbase["MUL_PROV_HY"] = entbase["PROV"] * entbase["HY"]
    entbase["MUL_PROV_ETYPE"] = entbase["PROV"] * entbase["ETYPE"]
    entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["INUM"]
    entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["ENUM"]
    entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["MPNUM"]
    entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["FSTINUM"]
    entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["TZINUM"]
#     entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["RGYEAR"]
#     entbase["MUL_PROV_INUM"] = entbase["PROV"] * entbase["ZCZB"]
    
    #new 12.1
    entbase["ZB_SUM"] = entbase["ZCZB"] + entbase["FINZB"]

    #12.10
    tmp = entbase.ZCZB.value_counts().reset_index()
    tmp.columns = ['ZCZB', 'ZCZB_COUNT']
    entbase = pd.merge(entbase, tmp, on="ZCZB", how="left")
    
    tmp = entbase.RGYEAR.value_counts().reset_index()
    tmp.columns = ['RGYEAR', 'RGYEAR_COUNT']
    entbase = pd.merge(entbase, tmp, on="RGYEAR", how="left")
    
    for column in ["MPNUM", "INUM", "FINZB", "FSTINUM", "TZINUM", "ENUM", "ZCZB", "INDEX_SUM", "RGYEAR"]:
#         groupby_list = [["HY"], ["ETYPE"], ["HY", "ETYPE"], ["HY", "PROV"], ["ETYPE", "PROV"]]
        groupby_list = [["HY"], ["ETYPE"], ["PROV"], ["HY", "PROV"], ["ETYPE", "PROV"], ["ZCZB_COUNT"], ["RGYEAR_COUNT"]]
        for groupby in groupby_list: # groupby是一个列表
            groupby_keylist = [] # groupby_keylist是一个列表
            for key in groupby:
                groupby_keylist.append(entbase[key]) # 具体地，groupby_keylist是一个 Series的列表
                    
			# 根据一个 series的列表 ，进行分组，然后计算每组的mean\min\max\sum
            tmp = entbase[column].groupby(groupby_keylist).mean().reset_index()
            tmp.columns = np.append(groupby, ["MEAN_COLUMN"])
            entbase = pd.merge(entbase, tmp, on=groupby, how="left")
            entbase["_".join(groupby) + "_" + column + "_MEAN_GAP"] = entbase[column] - entbase["MEAN_COLUMN"]
			# 原值与平均值的差距
            entbase["_".join(groupby) + "_" + column + "_RATE"] = entbase[column] / entbase["MEAN_COLUMN"]
			# 原值与平均值的比例
            entbase.drop(["MEAN_COLUMN"], axis=1, inplace=True)
            
            tmp = entbase[column].groupby(groupby_keylist).min().reset_index()
            tmp.columns = np.append(groupby, ["MIN_COLUMN"])
            entbase = pd.merge(entbase, tmp, on=groupby, how="left")
            entbase["_".join(groupby) + "_" + column + "_MIN_GAP"] = entbase[column] - entbase["MIN_COLUMN"]
			# 原值与最小值的差距
            entbase.drop(["MIN_COLUMN"], axis=1, inplace=True)
            
            tmp = entbase[column].groupby(groupby_keylist).max().reset_index()
            tmp.columns = np.append(groupby, ["MAX_COLUMN"])
            entbase = pd.merge(entbase, tmp, on=groupby, how="left")
            entbase["_".join(groupby) + "_" + column + "_MAX_GAP"] = entbase[column] - entbase["MAX_COLUMN"]
			# 原值与最大值的差距
            entbase.drop(["MAX_COLUMN"], axis=1, inplace=True)
            
            tmp = entbase[column].groupby(groupby_keylist).sum().reset_index()
            tmp.columns = np.append(groupby, ["SUM_COLUMN"])
            entbase = pd.merge(entbase, tmp, on=groupby, how="left")
            entbase["_".join(groupby) + "_" + column + "_SUM_GAP"] = entbase[column] / entbase["SUM_COLUMN"]
			# 原值占总数的比值
            entbase.drop(["SUM_COLUMN"], axis=1, inplace=True)
        
    entbase.drop(['RGYEAR_COUNT', 'ZCZB_COUNT'], axis = 1, inplace = True)
    #new 12.7
#     def divided_year(x) :
#         if x < 10:
#             return 1
#         if x < 20:
#             return 2
#         if x < 50:
#             return 3
#         if x < 100:
#             return 4
#         if x < 200:
#             return 5
#         if x < 500:
#             return 6
#         if  x < 1000:
#             return 7
#         if  x < 2000:
#             return 8
#         if  x < 5000:
#             return 9
#         if  x < 100000:
#             return 10
#         else:
#             return 11
    
#     entbase['FSTINUM_MUL_LOGZCZB'] = entbase['FSTINUM'] * entbase['ZCZB'].apply(lambda x : math.log(x + 1))
#     entbase['RGYEAR_MUL_ZCZB'] = entbase['RGYEAR'] * entbase['ZCZB']
#     entbase['RGYEAR_MUL_ZCZB'] = entbase.RGYEAR_MUL_ZCZB.apply(divided_year)
    
#     tmp = entbase['FSTINUM_MUL_LOGZCZB'].groupby(entbase.HY).agg(np.mean).reset_index()
#     tmp.columns = ["HY", "FSTINUM_MUL_LOGZCZB_HY_MEAN"]
#     entbase = pd.merge(entbase, tmp, on="HY", how="left")
#     tmp = entbase['FSTINUM_MUL_LOGZCZB'].groupby(entbase.ETYPE).agg(np.mean).reset_index()
#     tmp.columns = ["ETYPE", "FSTINUM_MUL_LOGZCZB_ETYPE_MEAN"]
#     entbase = pd.merge(entbase, tmp, on="ETYPE", how="left")
#     tmp = entbase['FSTINUM_MUL_LOGZCZB'].groupby(entbase.PROV).agg(np.mean).reset_index()
#     tmp.columns = ["PROV", "FSTINUM_MUL_LOGZCZB_PROV_MEAN"]
#     entbase = pd.merge(entbase, tmp, on="PROV", how="left")
#     entbase['FSTINUM_MUL_LOGZCZB_HY_GAP'] = entbase['FSTINUM_MUL_LOGZCZB'] - entbase["FSTINUM_MUL_LOGZCZB_HY_MEAN"]
#     entbase['FSTINUM_MUL_LOGZCZB_ETYPE_GAP'] = entbase['FSTINUM_MUL_LOGZCZB'] - entbase["FSTINUM_MUL_LOGZCZB_ETYPE_MEAN"]
#     entbase['FSTINUM_MUL_LOGZCZB_PROV_GAP'] = entbase['FSTINUM_MUL_LOGZCZB'] - entbase["FSTINUM_MUL_LOGZCZB_PROV_MEAN"]
    
#     tmp = entbase['MPNUM'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_MPNUM_MEAN", 'RGYEAR_MUL_ZCZB_MPNUM_STD']
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     tmp = entbase['INUM'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_INUM_MEAN", 'RGYEAR_MUL_ZCZB_INUM_STD']
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     tmp = entbase['FINZB'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_FINZB_MEAN", "RGYEAR_MUL_ZCZB_FINZB_STD"]
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     tmp = entbase['FSTINUM'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_FSTINUM_MEAN", "RGYEAR_MUL_ZCZB_FSTINUM_STD"]
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     tmp = entbase['TZINUM'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_TZINUM_MEAN", "RGYEAR_MUL_ZCZB_TZINUM_STD"]
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     tmp = entbase['ENUM'].groupby(entbase.RGYEAR_MUL_ZCZB).agg([np.mean, np.std]).reset_index()
#     tmp.columns = ["RGYEAR_MUL_ZCZB", "RGYEAR_MUL_ZCZB_ENUM_MEAN","RGYEAR_MUL_ZCZB_ENUM_STD"]
#     entbase = pd.merge(entbase, tmp, on="RGYEAR_MUL_ZCZB", how="left")
    
#     entbase["RGYEAR_MUL_ZCZB_MPNUM_GAP"] = entbase["MPNUM"] - entbase["RGYEAR_MUL_ZCZB_MPNUM_MEAN"]
#     entbase["RGYEAR_MUL_ZCZB_INUM_GAP"] = entbase['INUM'] - entbase["RGYEAR_MUL_ZCZB_INUM_MEAN"]
#     entbase["RGYEAR_MUL_ZCZB_FINZB_GAP"] = entbase['FINZB'] - entbase["RGYEAR_MUL_ZCZB_FINZB_MEAN"]
#     entbase["RGYEAR_MUL_ZCZB_FSTINUM_GAP"] = entbase["FSTINUM"] - entbase["RGYEAR_MUL_ZCZB_FSTINUM_MEAN"]
#     entbase["RGYEAR_MUL_ZCZB_TZINUM_GAP"] = entbase["TZINUM"] - entbase["RGYEAR_MUL_ZCZB_TZINUM_MEAN"]
#     entbase["RGYEAR_MUL_ZCZB_ENUM_GAP"] = entbase["ENUM"] - entbase["RGYEAR_MUL_ZCZB_ENUM_MEAN"]
        
    #new 12.6
    #entbase = entbase.join(pd.get_dummies(entbase["HY"], prefix='HY'))
#     entbase = entbase.join(pd.get_dummies(entbase["ETYPE"], prefix='ETYPE'))
#    entbase = entbase.join(pd.get_dummies(entbase["PROV"], prefix='PROV'))
#     entbase.drop(["HY", "ETYPE", "PROV"], axis=1, inplace=True)
#    entbase = entbase.join(pd.get_dummies(entbase["HY"], prefix='HY_RGYEAR').multiply(entbase["RGYEAR"], axis="index"))
#    entbase = entbase.join(pd.get_dummies(entbase["HY"], prefix='HY_ZCZB').multiply(entbase["ZCZB"], axis="index"))
#    entbase = entbase.join(pd.get_dummies(entbase["ETYPE"], prefix='ETYPE_RGYEAR').multiply(entbase["RGYEAR"], axis="index"))
#    entbase = entbase.join(pd.get_dummies(entbase["ETYPE"], prefix='ETYPE_ZCZB').multiply(entbase["ZCZB"], axis="index"))
#     entbase = entbase.join(pd.get_dummies(entbase["PROV"], prefix='PROV_ZCZB').multiply(entbase["ZCZB"], axis="index"))
#     entbase.drop(["HY", "ETYPE", "PROV"], axis=1, inplace=True)
#     entbase = entbase.join(pd.get_dummies(entbase["HY"], prefix='HY_RGYEAR_ZCZB').multiply(entbase["RGYEAR_ZCZB"], axis="index"))
#     entbase.drop(["PROV", "HY", "ETYPE"], axis=1, inplace=True)
    return entbase

entbase_feat = get_entbase_feature()
entbase_feat.head()


# In[5]:



def translate_money(x):
    if isinstance(x, float): return x
    if x == -1:
        return np.nan
    elif x[-18:] == ' (单位：万元)':
        return float(x[:-18])
    elif x[-9:] == '万美元':
        return float(x[:-9])
    elif x[-3:] == '万':
        return float(x[:-3])
    elif x[-12:] == '万人民币':
        return float(x[:-12])
    elif (x[-6:] == '万元'):
        if x[:-6] == 'null':
            return np.nan
        elif x[:9] == '人民币':
            return float(x[9:-6])
        elif (x[:6] == '美元') | (x[:6] == '港币'):
            return float(x[6:-6]) 
        else:
            return float(x[:-6])
    elif x[-9:] == '万港元':
        return float(x[:-9])
    else:
        return float(x)

def get_alter_feature():
    alter = alter_df.copy()
    
    alter["ALTDATE"] = alter["ALTDATE"].apply(translate_year)
    alter["ALTAF"] = alter["ALTAF"].apply(translate_money)
    alter["ALTBE"] = alter["ALTBE"].apply(translate_money)
    alter["ALT_BE_AF_GAP"] = alter["ALTAF"] - alter["ALTBE"] # 时间间隔
    
    alter_feat = alter["ALTERNO"].groupby(alter.EID).count().reset_index()
    alter_feat.columns = ["EID", "ALTER_COUNT"]
	# 1.每个企业有多少条记录
    
    tmp = alter["ALTERNO"].groupby(alter.EID).nunique().reset_index()
	# 2.每个企业有多少条不同变更类型的记录
    tmp.columns = ["EID", "ALTER_UNIQUE_COUNT"]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
    tmp = alter["ALTERNO"].groupby([alter.EID, alter.ALTERNO]).count().unstack().reset_index()#.fillna(0)
	# 3.找出每个企业每种变更类型的数量，一共有13种类型，所以多了13个特征！
	# tmp.columns现在是 eid, 加上各种变更类型的编码
    tmp.columns = [i if i == 'EID' else 'ALTERNO_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on='EID', how='left')
    for column in tmp.columns:
        if column != "EID":
            alter_feat[column + "_RATE"] = alter_feat[column] / alter_feat["ALTER_COUNT"]
    # 4.再多13个特征，分别是刚刚那些特征 占 该企业记录条数的 比值 
	
	
    tmp = alter["ALTDATE"].groupby(alter.EID).agg([min, max, np.ptp, np.std]).reset_index()
	# 5.以企业id进行分组，计算每一组的 altdate的min max ptp std 
    tmp.columns = [i if i == 'EID' else 'ALTDATE_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
   
    tmp = alter[alter.ALTERNO == "05"]["ALTBE"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    # 6.把alter为05号类型的那些找出来，根据企业id分组，每组计算ALTBE的mmps
	tmp.columns = [i if i == 'EID' else 'ALTBE_05_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 7.把alter为05号类型的那些找出来，根据企业id分组，每组计算ALTAF的mmps
    tmp = alter[alter.ALTERNO == "05"]["ALTAF"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALTAF_05_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 8.alter27号类型的
    tmp = alter[alter.ALTERNO == "27"]["ALTBE"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALTBE_27_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 8.alter27号类型的
    tmp = alter[alter.ALTERNO == "27"]["ALTAF"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALTAF_27_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 8.alter27或5号类型的
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTBE"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALTBE_05_27_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 8.alter27或5号类型的
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTAF"].groupby(alter.EID).agg([min, max, np.mean, np.std, np.ptp]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALTAF_05_27_'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
	# 下面是对  27   5  27或5   三者的gap year进行统计
    tmp = alter[alter.ALTERNO == "05"]["ALT_BE_AF_GAP"].groupby(alter.EID).agg([min ,max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALT_05_BE_AF_GAP'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
    tmp = alter[alter.ALTERNO == "27"]["ALT_BE_AF_GAP"].groupby(alter.EID).agg([min ,max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALT_27_BE_AF_GAP'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALT_BE_AF_GAP"].groupby(alter.EID).agg([min ,max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'ALT_05_27_BE_AF_GAP'+str(i) for i in tmp.columns]
    alter_feat = pd.merge(alter_feat, tmp, on="EID", how="left")
    
    #####################################################################################################
    #11.29
	
	# 下面这个跟上面差不多，不过分组的依据不是企业ID而是行业大类
    alter = pd.merge(alter, entbase_df[["EID", "HY"]], on="EID", how="left")
    alter_feat = pd.merge(alter_feat, entbase_df[["EID", "HY"]], on="EID", how="left")
    
    tmp = alter[alter.ALTERNO == "05"]["ALTBE"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_05_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_05_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_05_mean"] - alter_feat["HY_05_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["HY_05_ALTBE_MEAN"], axis=1)
    
    tmp = alter[alter.ALTERNO == "05"]["ALTAF"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_05_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_05_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_05_mean"] - alter_feat["HY_05_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["HY_05_ALTAF_MEAN"], axis=1)
    
    tmp = alter[alter.ALTERNO == "27"]["ALTBE"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_27_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_27_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_27_mean"] - alter_feat["HY_27_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["HY_27_ALTBE_MEAN"], axis=1)
    
	#                            这里写错了，应该是27
    tmp = alter[alter.ALTERNO == "05"]["ALTAF"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_27_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_27_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_27_mean"] - alter_feat["HY_27_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["HY_27_ALTAF_MEAN"], axis=1)
    
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTBE"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_05_27_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_05_27_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_05_27_mean"] - alter_feat["HY_05_27_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["HY_05_27_ALTBE_MEAN"], axis=1)
    
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTAF"].groupby(alter.HY).mean().reset_index()
    tmp.columns = ["HY", "HY_05_27_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="HY", how="left")
    alter_feat["HY_05_27_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_05_27_mean"] - alter_feat["HY_05_27_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["HY_05_27_ALTAF_MEAN"], axis=1)
    
    alter_feat = alter_feat.drop(["HY"], axis=1)
    #####################################################################################################
    
	
	# 嗯，分组的依据变为了PROV
    alter = pd.merge(alter, entbase_df[["EID", "PROV"]], on="EID", how="left")
    alter_feat = pd.merge(alter_feat, entbase_df[["EID", "PROV"]], on="EID", how="left")
    
    tmp = alter[alter.ALTERNO == "05"]["ALTBE"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_05_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_05_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_05_mean"] - alter_feat["PROV_05_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["PROV_05_ALTBE_MEAN"], axis=1)
    
    tmp = alter[alter.ALTERNO == "05"]["ALTAF"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_05_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_05_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_05_mean"] - alter_feat["PROV_05_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["PROV_05_ALTAF_MEAN"], axis=1)
    
    tmp = alter[alter.ALTERNO == "27"]["ALTBE"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_27_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_27_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_27_mean"] - alter_feat["PROV_27_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["PROV_27_ALTBE_MEAN"], axis=1)
    
    tmp = alter[alter.ALTERNO == "05"]["ALTAF"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_27_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_27_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_27_mean"] - alter_feat["PROV_27_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["PROV_27_ALTAF_MEAN"], axis=1)
    
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTBE"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_05_27_ALTBE_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_05_27_ALTBE_MEAN_GAP"] = alter_feat["ALTBE_05_27_mean"] - alter_feat["PROV_05_27_ALTBE_MEAN"]
    alter_feat = alter_feat.drop(["PROV_05_27_ALTBE_MEAN"], axis=1)
    
    tmp = alter[(alter.ALTERNO == "05") | (alter.ALTERNO == "27")]["ALTAF"].groupby(alter.PROV).mean().reset_index()
    tmp.columns = ["PROV", "PROV_05_27_ALTAF_MEAN"]
    alter_feat = pd.merge(alter_feat, tmp, on="PROV", how="left")
    alter_feat["PROV_05_27_ALTAF_MEAN_GAP"] = alter_feat["ALTAF_05_27_mean"] - alter_feat["PROV_05_27_ALTAF_MEAN"]
    alter_feat = alter_feat.drop(["PROV_05_27_ALTAF_MEAN"], axis=1)
    
    alter_feat = alter_feat.drop(["PROV"], axis=1)
    
    
    return alter_feat

alter_feat = get_alter_feature()
alter_feat.head()


# In[6]:


def get_branch_feature():
    branch = branch_df.copy()
    
    branch["BRANCH_LIVE_GAP"] = branch["B_ENDYEAR"] - branch["B_REYEAR"] 
	# 时间间隔
    
    branch_feat = branch["TYPECODE"].groupby(branch.EID).count().reset_index()
	# 每一个企业有多少分支 统计一下
    branch_feat.columns = ["EID", "BRANCH_COUNT"]
    
    tmp = branch['IFHOME'].groupby(branch.EID).sum().reset_index()
	# 在外省的分支的数目
    tmp.columns = ["EID", "BRANCH_HOME_COUNT"]
    branch_feat = pd.merge(branch_feat, tmp, on="EID", how="left")
    
    branch_feat["HOME_RATE"] = branch_feat["BRANCH_HOME_COUNT"] / branch_feat["BRANCH_COUNT"]
	# 在外省的分支的数目占分支总数的比例
    
	# 按企业ID进行分组，对于每一组，计算 分支成立、关停年度的min max ptp std 
    tmp = branch["B_REYEAR"].groupby(branch.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'BRANCH_REYEAR_'+str(i) for i in tmp.columns]
    branch_feat = pd.merge(branch_feat, tmp, on="EID", how="left")
    
    tmp = branch["B_ENDYEAR"].groupby(branch.EID).agg([min, max, np.ptp, np.std]).reset_index() 
	# 这些聚合函数计算的时候会忽略nan
    tmp.columns = [i if i == 'EID' else 'BRANCH_ENDYEAR_'+str(i) for i in tmp.columns]
    branch_feat = pd.merge(branch_feat, tmp, on="EID", how="left")
    
    tmp = branch["B_ENDYEAR"].groupby(branch.EID).count().reset_index()
	# 数一下每个企业有多少分支被关停了，以及被关停的占总数的比例
    tmp.columns = ["EID", "BRANCH_CLOSE_COUNT"]
    branch_feat = pd.merge(branch_feat, tmp, on="EID", how="left")
    
    branch_feat["CLOSE_RATE"] = branch_feat["BRANCH_CLOSE_COUNT"] / branch_feat["BRANCH_COUNT"]
    
	# 对于每一个企业，计算那些关停的企业存活的时间长度 的min max mean 
    tmp = branch[branch.B_ENDYEAR.notnull()]["BRANCH_LIVE_GAP"].groupby(branch.EID).agg([min, max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'BRANCH_LIVE_GAP_'+str(i) for i in tmp.columns]
    branch_feat = pd.merge(branch_feat, tmp, on="EID", how="left")
    
    return branch_feat

branch_feat = get_branch_feature()
branch_feat.head()


# In[7]:


def get_invest_feature():
    invest = invest_df.copy()
    
    invest["INVEST_LIVE_GAP"] = invest["BTENDYEAR"] - invest["BTYEAR"]
	# 时间间隔 
    
	# 每个企业分别获得了多少企业的投资
	# 这个特征应该叫 INVESTED_COUNT才对
    invest_feat = invest["BTEID"].groupby(invest.EID).count().reset_index()
    invest_feat.columns = ["EID", "INVEST_COUNT"]
    
	# 每个企业有多少来自外省的投资，及占的比重
    tmp = invest["IFHOME"].groupby(invest.EID).sum().reset_index()
    tmp.columns = ["EID", "INVEST_HOME_COUNT"]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    invest_feat["INVEST_HOME_RATE"] = invest_feat["INVEST_HOME_COUNT"] / invest_feat["INVEST_HOME_COUNT"]
    
	# 每个企业，投它的企业的持股比例、成立年度、关停年度的mmps
    tmp = invest["BTBL"].groupby(invest.EID).agg([min, max, np.mean, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'INVEST_BTBL_'+str(i) for i in tmp.columns]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
	
    tmp = invest["BTYEAR"].groupby(invest.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'INVEST_BTYEAR_'+str(i) for i in tmp.columns]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    tmp = invest["BTENDYEAR"].groupby(invest.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'INVEST_BTENDYEAR_'+str(i) for i in tmp.columns]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
	# 有多少被关停，占多少比例
    tmp = invest["BTENDYEAR"].groupby(invest.EID).count().reset_index()
    tmp.columns = ["EID", "INVEST_CLOSE_COUNT"]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    invest_feat["INVEST_CLOSE_RATE"] = invest_feat["INVEST_CLOSE_COUNT"] / invest_feat["INVEST_COUNT"]
    
	# 被关停的那些，存活的时间长度的统计
    tmp = invest[invest.BTENDYEAR.notnull()]["INVEST_LIVE_GAP"].groupby(invest.EID).agg([min, max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'INVEST_LIVE_GAP_'+str(i) for i in tmp.columns]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    #####################################################################################
    # 将公司按照投资人进行分组？？
	# ？？？不管了！
    tmp = invest["EID"].groupby(invest.BTEID).count().reset_index()
    tmp.columns = ["EID", "INVESTED_COUNT"]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
	# 有多少在省外的，以及比例
    tmp = invest["IFHOME"].groupby(invest.BTEID).sum().reset_index()
    tmp.columns = ["EID", "INVESTED_HOME_COUNT"]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    invest_feat["INVESTED_HOME_RATE"] = invest_feat["INVESTED_HOME_COUNT"] / invest_feat["INVESTED_HOME_COUNT"]
    
	
	##？？
    tmp = invest["BTBL"].groupby(invest.BTEID).agg([min, max, np.mean, np.ptp, np.std]).reset_index()
    tmp.columns = ["EID" if i == 'BTEID' else 'INVESTED_BTBL_'+str(i) for i in tmp.columns]
    invest_feat = pd.merge(invest_feat, tmp, on="EID", how="left")
    
    return invest_feat

invest_feat = get_invest_feature()
invest_feat.head()


# In[8]:


def get_right_feature():
    right = right_df.copy()
    
    right["ASKDATE"] = right["ASKDATE"].apply(translate_year)
    right["FBDATE"] = right[right.FBDATE.notnull()]["FBDATE"].apply(translate_year)
    right["RIGHT_ASK_FB_GAP"] = right["FBDATE"] - right["ASKDATE"]
	# 时间间隔

	# 按企业分组，统计权利ID个数
    right_feat = right["TYPECODE"].groupby(right.EID).count().reset_index()
    right_feat.columns = ["EID", "RIGHT_COUNT"]
    
	# 按（企业、权利类型）分组，统计权利type个数？？
    tmp = right["RIGHTTYPE"].groupby([right.EID, right.RIGHTTYPE]).count().unstack().reset_index()#.fillna(0)
    tmp.columns = [i if i == 'EID' else 'RIGHTTYPE_'+str(i) for i in tmp.columns]
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    for column in tmp.columns:
        if column != "EID":
            right_feat[column + "_RATE"] = right_feat[column] / right_feat["RIGHT_COUNT"]
    
	# 下面套路差不多
    tmp = right["ASKDATE"].groupby(right.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'RIGHT_ASKDATE_'+str(i) for i in tmp.columns]
    right_feat = pd.merge(right_feat, tmp, on="EID", how="left")
    
    tmp = right["FBDATE"].groupby(right.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'RIGHT_FBDATE_'+str(i) for i in tmp.columns]
    right_feat = pd.merge(right_feat, tmp, on="EID", how="left")
    
    tmp = right[right.FBDATE.notnull()]["RIGHT_ASK_FB_GAP"].groupby(right.EID).agg([min, max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'RIGHT_ASK_FB_GAP_'+str(i) for i in tmp.columns]
    right_feat = pd.merge(right_feat, tmp, on="EID", how="left")
    
    return right_feat

right_feat = get_right_feature()
right_feat.head()


# In[9]:


def get_project_feature():
    project = project_df.copy()

    project["DJDATE"] = project["DJDATE"].apply(translate_year)
    
    project_feat = project["TYPECODE"].groupby(project.EID).count().reset_index()
    project_feat.columns = ["EID", "PROJ_COUNT"]
    
    tmp = project["DJDATE"].groupby(project.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'PROJ_DJDATE_'+str(i) for i in tmp.columns]
    project_feat = pd.merge(project_feat, tmp, on="EID", how="left")
    
    tmp = project["IFHOME"].groupby(project.EID).sum().reset_index()
    tmp.columns = ["EID", "PROJ_HOME_COUNT"]
    project_feat = pd.merge(project_feat, tmp, on="EID", how="left")
    
    project_feat["PROJ_HOME_RATE"] = project_feat["PROJ_COUNT"] / project_feat["PROJ_HOME_COUNT"]
    
    return project_feat

project_feat = get_project_feature()
project_feat.head()


# In[10]:


def translate_date(x):
    x = x.replace('年','')
    x = x.replace('月', '')
    
    year = int(x[:4])
    month = int(x[4:])
    return (2015-year)*12 - month

def get_lawsuit_feature():
    lawsuit = lawsuit_df.copy()
    
    lawsuit["LAWDATE"] = lawsuit["LAWDATE"].apply(translate_date)
    
    lawsuit_feat = lawsuit["TYPECODE"].groupby(lawsuit.EID).count().reset_index()
    lawsuit_feat.columns = ["EID", "LAWSUIT_COUNT"]
    
    tmp = lawsuit["LAWDATE"].groupby(lawsuit.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'LAWDATE_'+str(i) for i in tmp.columns]
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on="EID", how="left")
    
    tmp = lawsuit["LAWAMOUNT"].groupby(lawsuit.EID).agg([sum, min, max, np.mean, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'LAWAMOUNT_'+str(i) for i in tmp.columns]
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on="EID", how="left")
    
    tmp = lawsuit["LAWAMOUNT"].groupby(lawsuit.EID).count().reset_index()
    tmp.columns = ["EID", "LAW_HAS_AMOUNT_COUNT"]
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on="EID", how="left")
    
    lawsuit_feat["LAW_HAS_AMOUNT_RATE"] = lawsuit_feat["LAW_HAS_AMOUNT_COUNT"] /  lawsuit_feat["LAWSUIT_COUNT"]
    
    return lawsuit_feat

lawsuit_feat = get_lawsuit_feature()
lawsuit_feat.head()


# In[11]:


def get_breakfaith_feature():
    breakfaith = breakfaith_df.copy()
    
    breakfaith["FBDATE"] = breakfaith["FBDATE"].apply(translate_date)
    breakfaith["SXENDDATE"] = breakfaith[breakfaith.SXENDDATE.notnull()]["SXENDDATE"].apply(translate_date)
    breakfaith["BREAKFAITH_SXDATE_GAP"] = breakfaith["SXENDDATE"] - breakfaith["FBDATE"]
    
    
    breakfaith_feat = breakfaith["TYPECODE"].groupby(breakfaith.EID).count().reset_index()
    breakfaith_feat.columns = ["EID", "BREAKFAITH_COUNT"]
    
    tmp = breakfaith["FBDATE"].groupby(breakfaith.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'FBDATE_'+str(i) for i in tmp.columns]
    breakfaith_feat = pd.merge(breakfaith_feat, tmp, on="EID", how="left")

    tmp = breakfaith["SXENDDATE"].groupby(breakfaith.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'SXENDDATE_'+str(i) for i in tmp.columns]
    breakfaith_feat = pd.merge(breakfaith_feat, tmp, on="EID", how="left")

    tmp = breakfaith["SXENDDATE"].groupby(breakfaith.EID).count().reset_index()
    tmp.columns = ["EID", "BREAKFAITH_SXEND_COUNT"]
    breakfaith_feat = pd.merge(breakfaith_feat, tmp, on="EID", how="left")
    
    breakfaith_feat["BREAKFAITH_SXEND_RATE"] = breakfaith_feat["BREAKFAITH_SXEND_COUNT"] / breakfaith_feat["BREAKFAITH_COUNT"]
    
    tmp = breakfaith[breakfaith.SXENDDATE.notnull()]["BREAKFAITH_SXDATE_GAP"].groupby(breakfaith.EID).agg([min, max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'BREAKFAITH_SXDATE_GAP_'+str(i) for i in tmp.columns]
    breakfaith_feat = pd.merge(breakfaith_feat, tmp, on="EID", how="left")
    
    return breakfaith_feat

breakfaith_feat = get_breakfaith_feature()
breakfaith_feat.head()


# In[12]:


def transform_pnum(x):
    if x == 0:
        return 0
    if x == '若干':
        return 1
    elif x[-3:] == '人':
        return float(x[:-3])
    else:
        return float(x)

def get_recruit_feature():
    recruit = recruit_df.copy()
    
    recruit.PNUM.fillna(0, inplace=True)
    recruit['RECR_YEAR_GAP'] = 2017 - pd.to_datetime(recruit.RECDATE).dt.year ####!!!!
    recruit["PNUM"] = recruit["PNUM"].apply(transform_pnum)
    recruit["RECDATE"] = recruit["RECDATE"].apply(translate_year)
    
    recruit_feat = recruit["PNUM"].groupby(recruit.EID).agg([len, sum]).reset_index()
    recruit_feat.columns = ["EID", "RECRUIT_TIMES", "RECRUIT_COUNT"]
    
    tmp = recruit["RECDATE"].groupby(recruit.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'RECDATE_'+str(i) for i in tmp.columns]
    recruit_feat = pd.merge(recruit_feat, tmp, on="EID", how="left")
    
    ####!!!!
	tmp = recruit[(recruit.RECR_YEAR_GAP != 9) & (recruit.RECR_YEAR_GAP != 5)].groupby(["EID", "RECR_YEAR_GAP"])["PNUM"].sum().unstack().reset_index()
    tmp["4_BI_3"] = (tmp[3] - tmp[4]) / tmp[4]
    tmp["3_BI_2"] = (tmp[2] - tmp[3]) / tmp[3]
    tmp.drop([2, 3, 4], axis=1, inplace=True)
    tmp.columns = (i if i == 'EID' else 'RECRUIT_INCREATE_RATE_'+ str(i) for i in tmp.columns)
    recruit_feat = pd.merge(recruit_feat, tmp, on="EID", how="left")
    
    tmp = recruit.groupby(['EID', 'WZCODE'])['PNUM'].sum().unstack().reset_index()
    tmp.columns = [i if i == 'EID' else 'WZCODE_'+str(i) for i in tmp.columns]
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
    
    return recruit_feat

recruit_feat = get_recruit_feature()
recruit_feat.head()


# In[13]:


def to_month(x):
    x = x.decode("gbk")
    x = x.replace(u'年','')
    x = x.replace(u'月', '')
    
    year = int(x[:4])
    month = int(x[4:])
    return (2015-year)*12 - month
    
def get_qualification_feature():
    qual = qualification_df.copy()
        
    qual["BEGINDATE"] = qual["BEGINDATE"].apply(to_month)
    qual["EXPIRYDATE"] = qual[qual.EXPIRYDATE.notnull()]["EXPIRYDATE"].apply(to_month)
    qual["QUAL_DATE_GAP"] = qual["EXPIRYDATE"] - qual["BEGINDATE"]
    
    qual_feat = qual["ADDTYPE"].groupby(qual.EID).count().reset_index()
    qual_feat.columns = ["EID", "QUAL_COUNT"]
    
    tmp = qual["ADDTYPE"].groupby([qual.EID, qual.ADDTYPE]).count().unstack().reset_index()#.fillna(0)
    tmp.columns = [i if i == 'EID' else 'QUAL_ADDTYPE_'+str(i) for i in tmp.columns]
    qual_feat = pd.merge(qual_feat, tmp, on='EID', how='left')
    
    tmp = qual["BEGINDATE"].groupby(qual.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'QUAL_BEGINDATE_'+str(i) for i in tmp.columns]
    qual_feat = pd.merge(qual_feat, tmp, on="EID", how="left")
    
    tmp = qual["EXPIRYDATE"].groupby(qual.EID).agg([min, max, np.ptp, np.std]).reset_index()
    tmp.columns = [i if i == 'EID' else 'QUAL_EXPIRYDATE_'+str(i) for i in tmp.columns]
    qual_feat = pd.merge(qual_feat, tmp, on="EID", how="left")
    
    tmp = qual[qual["EXPIRYDATE"].notnull()]["QUAL_DATE_GAP"].groupby(qual.EID).agg([min, max, np.mean]).reset_index()
    tmp.columns = [i if i == 'EID' else 'QUAL_DATE_GAP_'+str(i) for i in tmp.columns]
    qual_feat = pd.merge(qual_feat, tmp, on="EID", how="left")
    
    tmp = qual["EXPIRYDATE"].groupby(qual.EID).count().reset_index()
    tmp.columns = ["EID", "QUAL_HAS_EXPIRYDATE_COUNT"]
    qual_feat = pd.merge(qual_feat, tmp, on="EID", how="left")
    
    qual_feat["QUAL_HAS_EXPIRYDATE_RATE"] = qual_feat["QUAL_HAS_EXPIRYDATE_COUNT"] / qual_feat["QUAL_COUNT"]
    
    return qual_feat

qual_feat = get_qualification_feature()
qual_feat.head()


# In[14]:


train = pd.read_csv("./data/train.csv")
labels = train.TARGET.values

test = pd.read_csv("./data/evaluation_public.csv")
train.head()


# In[15]:


train_feat = pd.merge(train[["EID"]], entbase_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, alter_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, branch_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, invest_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, right_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, project_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, lawsuit_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, breakfaith_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, recruit_feat, on="EID", how="left")
train_feat = pd.merge(train_feat, qual_feat, on="EID", how="left")

train_feat.head()


# In[16]:


test_feat = pd.merge(test, entbase_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, alter_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, branch_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, invest_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, right_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, project_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, lawsuit_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, breakfaith_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, recruit_feat, on="EID", how="left")
test_feat = pd.merge(test_feat, qual_feat, on="EID", how="left")

test_feat.head()


# In[17]:


train_count = train_feat.shape[0]
test_count = test_feat.shape[0]

# 所以下面是跨表之间的组合特征咯？
all_feat = pd.concat([train_feat, test_feat]).reset_index(drop=True)

all_feat["ALTER_COUNT_MEAN_YEAR"] = all_feat["ALTER_COUNT"] / all_feat["RGYEAR"]
all_feat["BRANCH_COUNT_MEAN_YEAR"] = all_feat["BRANCH_COUNT"] / all_feat["RGYEAR"]
all_feat["RIGHT_COUNT_MEAN_YEAR"] = all_feat["RIGHT_COUNT"] / all_feat["RGYEAR"]
all_feat["PROJ_COUNT_MEAN_YEAR"] = all_feat["PROJ_COUNT"] / all_feat["RGYEAR"]
all_feat["LAWSUIT_COUNT_MEAN_YEAR"] = all_feat["LAWSUIT_COUNT"] / all_feat["RGYEAR"]
all_feat["SX_COUNT_MEAN_YEAR"] = all_feat["BREAKFAITH_COUNT"] / all_feat["RGYEAR"]
all_feat["RECRUIT_COUNT_MEAN_YEAR"] = all_feat["RECRUIT_COUNT"] / all_feat["RGYEAR"]
all_feat["RECRUIT_TIMES_MEAN_YEAR"] = all_feat["RECRUIT_TIMES"] / all_feat["RGYEAR"]
all_feat["INDEX_SUM_MEAN_YEAR"] = all_feat["INDEX_SUM"] / all_feat["RGYEAR"]

# all_feat["ALTER_COUNT_MEAN_YEAR1"] = all_feat["ALTER_COUNT"] / all_feat["RGYEAR"]
# all_feat["BRANCH_COUNT_MEAN_YEAR1"] = all_feat["BRANCH_COUNT"] / all_feat["RGYEAR"]
# all_feat["RIGHT_COUNT_MEAN_YEAR1"] = all_feat["RIGHT_COUNT"] / all_feat["RGYEAR"]
# all_feat["PROJ_COUNT_MEAN_YEAR1"] = all_feat["PROJ_COUNT"] / all_feat["RGYEAR"]
# all_feat["LAWSUIT_COUNT_MEAN_YEAR1"] = all_feat["LAWSUIT_COUNT"] / all_feat["RGYEAR"]
# all_feat["SX_COUNT_MEAN_YEAR1"] = all_feat["BREAKFAITH_COUNT"] / all_feat["RGYEAR"]
# all_feat["RECRUIT_COUNT_MEAN_YEAR1"] = all_feat["RECRUIT_COUNT"] / all_feat["RGYEAR"]
# all_feat["RECRUIT_TIMES_MEAN_YEAR1"] = all_feat["RECRUIT_TIMES"] / all_feat["RGYEAR"]
# all_feat["INDEX_SUM_MEAN_YEAR1"] = all_feat["INDEX_SUM"] / all_feat["RGYEAR"]

all_feat.RIGHT_COUNT.fillna(0, inplace=True)
all_feat["RGYEAR_RIGHT_NUM"] = get_interaction_feature(all_feat, "RGYEAR", "RIGHT_COUNT")

all_feat.ALTER_COUNT.fillna(0, inplace=True)
all_feat["RGYEAR_ALTER_COUNT"] = get_interaction_feature(all_feat, "RGYEAR", "ALTER_COUNT")

all_feat.RECRUIT_COUNT.fillna(0, inplace=True)
all_feat["RGYEAR_RECRUIT_COUNT"] = get_interaction_feature(all_feat, "RGYEAR", "RECRUIT_COUNT")

# for column in ["ALTER_COUNT", "RIGHT_COUNT"]:

for column in ["ALTER_COUNT", "RIGHT_COUNT"]:
    #groupby_list = [["HY"], ["ETYPE"], ["HY", "ETYPE"], ["HY", "PROV"], ["ETYPE", "PROV"]]
    groupby_list = [["HY"], ["HY", "PROV"],["PROV"]]
    for groupby in groupby_list:
        groupby_keylist = []
        for key in groupby:
            groupby_keylist.append(all_feat[key])

        tmp = all_feat[column].groupby(groupby_keylist).mean().reset_index()
        tmp.columns = np.append(groupby, ["MEAN_COLUMN"])
        all_feat = pd.merge(all_feat, tmp, on=groupby, how="left")
        all_feat["_".join(groupby) + "_" + column + "_MEAN_GAP"] = all_feat[column] - all_feat["MEAN_COLUMN"]
    #             if column in ["ZCZB", "FINZB"]:
    #                 all_feat["_".join(groupby) + "_" + column + "_RATE"] = all_feat[column] / all_feat["MEAN_COLUMN"]
        all_feat.drop(["MEAN_COLUMN"], axis=1, inplace=True)

        tmp = all_feat[column].groupby(groupby_keylist).min().reset_index()
        tmp.columns = np.append(groupby, ["MIN_COLUMN"])
        all_feat = pd.merge(all_feat, tmp, on=groupby, how="left")
        all_feat["_".join(groupby) + "_" + column + "_MIN_GAP"] = all_feat[column] - all_feat["MIN_COLUMN"]
        all_feat.drop(["MIN_COLUMN"], axis=1, inplace=True)

        tmp = all_feat[column].groupby(groupby_keylist).max().reset_index()
        tmp.columns = np.append(groupby, ["MAX_COLUMN"])
        all_feat = pd.merge(all_feat, tmp, on=groupby, how="left")
        all_feat["_".join(groupby) + "_" + column + "_MAX_GAP"] = all_feat[column] - all_feat["MAX_COLUMN"]
        all_feat.drop(["MAX_COLUMN"], axis=1, inplace=True)

        tmp = all_feat[column].groupby(groupby_keylist).sum().reset_index()
        tmp.columns = np.append(groupby, ["SUM_COLUMN"])
        all_feat = pd.merge(all_feat, tmp, on=groupby, how="left")
        all_feat["_".join(groupby) + "_" + column + "_SUM_GAP"] = all_feat[column] / all_feat["SUM_COLUMN"]
        all_feat.drop(["SUM_COLUMN"], axis=1, inplace=True)
        
#all_feat.drop(["HY", "ETYPE", "PROV"], axis=1, inplace=True)


# In[18]:


train_feat = all_feat[:train_count].reset_index(drop=True)
test_feat = all_feat[-test_count:].reset_index(drop=True)

# enc = ce.LeaveOneOutEncoder(cols=["HY", "ETYPE"])
# enc.fit(train_feat, labels)
# train_feat = enc.transform(train_feat)
# test_feat = enc.transform(test_feat)

train_feat.head()


# In[19]:


#train_feat["TARGET"] = labels
#train_feat.to_csv("./trainFeature_jj.csv", index=False)
#test_feat.to_csv("./testFeature_jj.csv", index=False)
#train_feat.drop(["TARGET"], axis=1, inplace=True)

test_feat['EID'] = test_feat["EID"].apply(lambda x: int(x[1:]))
train_feat['EID'] = train_feat["EID"].apply(lambda x: int(x[1:]))


# In[51]:


import time

params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    
    'max_depth': 10,
    'subsample': 0.7, #0.7
    'colsample_bytree': 0.7, #0.7,
    'stratified':True,
    'min_child_weight': 15,
    'eta': 0.01,
    'seed': 20,
    'silent':1
}

dtrain = xgb.DMatrix(train_feat, label=labels)
dtest = xgb.DMatrix(test_feat, label=np.zeros(test_feat.shape[0]))

print "start cv:", time.strftime("%H:%M:%S",time.localtime())
res = xgb.cv(params, dtrain, 2000, nfold=5, early_stopping_rounds=50, verbose_eval=10)
print "done cv:", time.strftime("%H:%M:%S",time.localtime())
print "best cv:", res['test-auc-mean'].tail(1).values[0]


# In[ ]:


watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, len(res), watchlist, verbose_eval=10)


# In[ ]:


def plot_feature_importance(model):
    df = pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
    print "feature， importance"
    sum = df['importance'].sum()
    for index, row in df.iterrows():
        print row["feature"], row["importance"]
plot_feature_importance(model)


# In[ ]:


ptest = model.predict(dtest)
sub = pd.DataFrame({'EID':test.EID, 'FORTARGET':[1 if i > 0.18 else 0 for i in ptest], 'PROB':ptest})
sub.to_csv('./1209-0.6892068' + ".csv", index=0)

