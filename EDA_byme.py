# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:03:12 2019

@author: Administrator
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile#卡方检验,卡方分布
from scipy import stats
import datetime,time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import gc
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


path='C:/Users/Administrator/Desktop/kedaxunfei'
test = pd.read_table(f'{path}/round1_iflyad_test_feature.txt',index_col='instance_id')
train = pd.read_table(f'{path}/round1_iflyad_train.txt',index_col='instance_id')
data = pd.concat([train, test], axis=0, ignore_index=True)#拼凑上下
'''
基本数据:
instance_id 样本id
click 是否点击

广告信息:
adid 广告id
advert_id 广告主id
orderid 订单id
advert_industry_inner 广告主行业
advert_name 广告主名称
campaign_id 活动id
creative_id 创意id
creative_type 创意类型
creative_tp_dnf 样式定向id
creative_is_jump 是否是落地页跳转
creative_is_download 是否是落地页下载
creative_is_js 是否为js素材
creative_is_voicead 是否是语音广告
creative_width 创意宽
creative_height 创意高
creative_has_deeplink 响应素材是否有deeplink

媒体信息：
app_cate_id app分类
f_channel 一级频道
app_id 媒体id
inner_slot_id 媒体广告位
app_paid app是否付费

用户信息：
user_tags 用户标签信息，以逗号分隔

上下文信息：
city 城市
carrier 运行商
time 时间戳
province 省份
nnt 联网类型
devtype 设备类型
os_name 操作系统名称
osv 操作系统版本
os 操作系统
make 品牌
model 机型
'''
#于CTR问题而言，广告是否被点击的主导因素是用户，其次是广告信息。
#所以我们要做的是充分挖掘用户及用户行为信息，然后才是广告主、广告等信息。

#每个样本应该是一次广告的一次投放对应的一次曝光的点击与否的情况

# 检查两表的userID是否有交集
print(len(np.intersect1d(train.index,test.index)))
#无广告交集
train.info()
#查看缺失特征：user_tags，make，model，osv，app_cate_id，f_channel，app_id
train.nunique()#查看有没有单个值的特征：只有creative_is_js 是否为js素材 creative_is_voicead 是否是语音广告，app_paid，考虑剔除

train['click'].sum()/(train['click'].count()-train['click'].sum())
#训练集正负样本比例,约1：4

#查看每个特征方便清洗和特征构造
#%%
#adid 广告id
len(train['adid'].unique())#2079个广告
train['adid'].size#1001650条记录

id_size = train.groupby('adid').size().sort_values()#每条广告的记录数
id_size.describe()#平均每个广告481条记录
plt.boxplot(id_size)#箱线图
plt.hist(id_size, bins=40, alpha=0.75,label="")#频数直方图
id_size.value_counts()#广告记录数对应广告量
plt.scatter(id_size.value_counts().index,id_size.value_counts().values)#散点图

id_test=train.groupby('adid').nunique()
#查看与广告id的对应关系，一个广告id唯一对应：advert_id,orderid，campaign_id,creative_id,creative_tp_dnf,creative_type,creative_width,creative_height等具体广告特征,还有advert_name

#取一个广告观察数据
id_test1=train[train['adid']==train['adid'].unique()[0]]
id_test1.nunique()#重复率最低的特征是时间和user_tags，其他都很高
id_test1.info()
len(id_test1)
#有重复有不重复，有特征字段缺失数据
id_test1.groupby('user_tags').size().sort_values().max()#一个用户最高有3171次记录
id_test1.groupby('adid')['time'].apply(lambda x:x.value_counts().iloc[0]).sort_values().max()#最大时间重复数为4
#%%
#advert 广告主id
len(train['advert_id'].unique())#38个广告主
train['advert_id'].size#1001650条记录
train.groupby('advert_id').size()
train.groupby('advert_id')['click'].apply(lambda x: x.sum()/x.count())
(train.groupby('advert_id').size()).corr(train.groupby('advert_id')['click'].apply(lambda x: x.sum()/x.count()))
#0.42低度相关,由于每个广告主的广告数量不同，查看数量与点击率的相关性，说明跟数量关系不大
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('advert_id')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('advert_id')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
#P＜0.05，差异有显著统计学意义
#%%
#orderid订单id
print('订单id有%s个' % len(train['orderid'].unique()))#936个订单id
order_test = train.groupby('orderid').nunique()#orderid对应每个特征的不同值数
order_test['count'] = train.groupby('orderid').size()
print(order_test.count()[order_test.max()==1].index)#查看与orderid一一对应的特征
'''
订单id与
advert_id广告主id,orderid,campaign_id活动id,creative_type创意类型,creative_is_jump是否是落地页跳转,creative_is_download是否是落地页下载
creative_is_js是否为js素材,creative_is_voicead是否是语音广告,creative_has_deeplink响应素材是否有deeplink,app_paidapp是否付费,advert_name广告主名称
这些特征是一个订单id唯一对应一个特征变量
与adid广告id不对应，因为order_test['adid'].describe()里中位数是2
检查异常值的问题：
advert_industry_inner 广告主行业，creative_tp_dnf 样式定向id，creative_width 创意宽
creative_height 创意高
其他信息问题不大
'''
#由于订单id对应唯一的'campaign_id', 'creative_type'，'advert_id'，我们取'campaign_id'和'advert_id'就能代替
#%%
#time
train.groupby('adid')['time'].apply(lambda x:x.value_counts().iloc[0]).sort_values()#最大重复数
#最大重复时间为24
train['time'].unique()
len(train['time'].unique())
train['hour'] = train['time'].apply(lambda x:int(time.strftime("%H", time.localtime(x))))
X_gender = train['hour'].value_counts().sort_index().index#Y标签值
Y_gender = train['hour'].value_counts().sort_index()#Y标签值
#plt.bar(X_gender,Y_gender, width=0.8,label="" )  
Y_gender_0 = train[train['click']==0]['hour'].value_counts().sort_index()
Y_gender_1 = train[train['click']==1]['hour'].value_counts().sort_index()
plt.bar(X_gender,Y_gender_0, alpha=0.75, width=0.8)
plt.bar(X_gender,Y_gender_1, alpha=0.75, width=0.8, bottom=Y_gender_0)  # 通过 bottom=Y_gender_0 设置柱叠加 ，堆叠图

#%%
#campaign_id 活动id
len(train['campaign_id'].unique())#64个
train['campaign_id'].unique()
train.groupby('campaign_id')['click'].apply(lambda x: x.sum()/x.count())
campaign_id_test = train.groupby('campaign_id').nunique()
#与advert_id唯一对应
#%%
#creative_id 创意id
len(train['creative_id'].unique())#862个
train['creative_id'].unique()
train.groupby('creative_id')['click'].apply(lambda x: x.sum()/x.count())
creative_id_test = train.groupby('creative_id').nunique()##与advert_id，creative_tp_dnf,creative_type,creative_width,creative_height,creative__has_deeplink,advert_name唯一对应
#%%
#creative_type 创意类型
print('创意类型有%s种，分别是%s' % (len(train['creative_type'].unique()),train['creative_type'].unique()))#5
train['creative_type'].unique()
train.groupby('creative_type')['click'].apply(lambda x: x.sum()/x.count())
train.groupby('creative_type').size()

#查看对应广告特征：creative_is_jump 是否是落地页跳转 creative_is_download 是否是落地页下载
#creative_is_js 是否为js素材 creative_is_voicead 是否是语音广告 creative_width 创意宽
#creative_height 创意高 creative_has_deeplink 响应素材是否有deeplink

#查看其他特征与对应广告特征的关系
df_type_test = train.groupby('creative_type').nunique()
#与创意类型一一对应的只有creative_is_js 是否为js素材 creative_is_voicead 是否是语音广告，app_paid,与活动id和创意id更是一对多
#%%
#advert_industry_inner 广告主行业
#上面有看见一个订单id对应了2个广告行业，查看分布
print('大行业有%s种，分别是%s' % (len(train['advert_industry_inner'].unique()),train['advert_industry_inner'].unique()))#大行业有24种
plt.hist(order_test['advert_industry_inner'], bins=2, alpha=0.75,label="")
len(order_test[order_test['advert_industry_inner']==2])#3个订单id对应了2个广告行业，应该是脏数据，考虑剔除
industry_inner_test = train.groupby('advert_industry_inner').nunique()
#与广告主行业一一对应的同样只有creative_is_js 是否为js素材 creative_is_voicead 是否是语音广告，app_paid
#%%
#advert_name 广告主名称
advert_name_test1 = train.groupby('advert_id').nunique()#id底下唯一对应一个名字，名字不唯一对应一个id
advert_name_test = train.groupby('advert_name').nunique()#所以一个广告主名字可能有多个id
print(advert_name_test.max()[advert_name_test.max()==1].index)
advert_name_test['advert_id']#advert_name 广告主名称与广告主id不是一一对应
train[['advert_id','advert_name']].drop_duplicates().set_index('advert_name',drop=True).loc[advert_name_test[advert_name_test['advert_id']!=1].index]
train[['advert_id','advert_name']].set_index('advert_name',drop=True).loc[advert_name_test[advert_name_test['advert_id']!=1].index]

#%%
#creative_width 创意宽
#creative_height 创意高
train['creative_width'].unique()
train['creative_height'].unique()
#离散数值
train.groupby('creative_width')['click'].apply(lambda x: x.sum()/x.count())
train.groupby('creative_height')['click'].apply(lambda x: x.sum()/x.count())
plt.scatter(train.groupby('creative_width')['click'].apply(lambda x: x.sum()/x.count()).index,train.groupby('creative_width')['click'].apply(lambda x: x.sum()/x.count()).values)#散点图
plt.scatter(train.groupby('creative_height')['click'].apply(lambda x: x.sum()/x.count()).index,train.groupby('creative_height')['click'].apply(lambda x: x.sum()/x.count()).values)#散点图
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('creative_width')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('creative_width')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('creative_height')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('creative_height')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
#创意宽，高对于点击率有显著的影响
#%%
#inner_slot_id 媒体广告位
len(train['inner_slot_id'].unique())#1169个广告位，而我们有2079个广告id
inner_slot_test = train.groupby('inner_slot_id').nunique()#没有一一对应值
#广告位的长宽是固定的吗
inner_slot_test['creative_width'].value_counts()
inner_slot_test['creative_height'].value_counts()
#不固定，但是不变的占多数，最多一个广告位有3个尺度
#%%
#app_cate_id app分类
#f_channel 一级频道
#app_id
print(len(train['app_cate_id'].unique()))#23个
print(len(train['f_channel'].unique()))#74个
print(len(train['app_id'].unique()))#439个
app_cate_test = train.groupby('app_cate_id').nunique()#与creative_has_deeplink 响应素材是否有deeplink唯一对应
app_id_test = train.groupby('app_id').nunique()#与app_cate_id和creative_has_deeplink一一对应
channel_test = train.groupby('f_channel').nunique()#'carrier'运行商, 'devtype'设备类型, 'app_cate_id'app分类, 'app_id'媒体id,'creative_is_jump', 'creative_is_download', 'creative_has_deeplink'唯一对应, make品牌全是0，即有一级频道的记录没有品牌数据
print(channel_test.max()[channel_test.max()==1].index)
train.groupby('app_cate_id')['click'].apply(lambda x: x.sum()/x.count())
train.groupby('f_channel')['click'].apply(lambda x: x.sum()/x.count())
'''
f_channel                76390 non-null object
app_id                   999383 non-null float64
由于f_channel下一个频道对应一个app_id，由于缺失值过多，考虑将f_channel去除，只使用app_id
'''
#%%
#model 机型
#make 品牌
print(len(train['model'].unique()))#14054个
print(len(train['make'].unique()))#3141个
#994248 non-null object/1001650
train['model'].count()/train['model'].size#99.2%的非缺失率
train['make'].count()/train['make'].size#90.1%的非缺失率
train['model'].value_counts()
train.groupby('model')['click'].apply(lambda x: x.sum()/x.count()).sort_values()
train.groupby('make')['click'].apply(lambda x: x.sum()/x.count()).sort_values()#很多的点击率是100%，明显是由于基数过大与过少，这样的数据不具有泛化能力
#手机品牌太多，粒度太细容易过拟合，考虑后续合并手机类型
#%%
#user_tags 用户标签信息，以逗号分隔
train['user_tags']
train[train['user_tags'].notnull()]['user_tags'].nunique()
train[train['user_tags'].notnull()]['click'].value_counts()#0    557869 1    134011
train[(train['user_tags'].notnull())&(train['click']==0)][['user_tags','click']]
#发现有用户点击也是0
train[train['user_tags'].isnull()]['click'].value_counts()#0    244994 1     64776
#用户缺失不影响点击与否
test['user_tags']
#用户标签信息理解为用户的属性和动作吧，后续用词向量的方式提取相关信息
#%%
#city 城市
#province 省份
train['city'].unique()
train['province'].unique()
print(len(train['city'].unique()))#333个 6——12位不同
print(len(train['province'].unique()))#35个，前缀后缀完全一样，只有6-9位不同
#一般city会有省市区等更多待挖掘信息，这样的特征会更有利于建模
#%%
#nnt 联网类型
train['nnt'].unique()#6种
train['nnt'].value_counts()
train.groupby('nnt')['click'].apply(lambda x: x.sum()/x.count())
(train.groupby('nnt').size()).corr(train.groupby('nnt')['click'].apply(lambda x: x.sum()/x.count()))
#数量与点击率没有太大关系
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('nnt')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('nnt')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
#单看点击率差异，我们需要排除数量的影响，于是用卡方检验
#%%
#devtype 设备类型
print(len(train['devtype'].unique()))#14054个
train['devtype'].unique()
devtype_test = train.groupby('devtype').nunique()
print(devtype_test.count()[devtype_test.max()==1].index)#查看与devtype唯一对应的特征，除去只有一个数值的变量，没有
train.groupby('devtype')['click'].apply(lambda x: x.sum()/x.count())
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('devtype')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('devtype')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
#%%
#os_name 操作系统名称
#字段无缺失值
print(len(train['os_name'].unique()))#14054个
train['os_name'].unique()
train.groupby('os_name')['click'].apply(lambda x: x.sum()/x.count())#区别不大，但未知操作系统的显著
stats.chi2_contingency(
        pd.DataFrame({'a':train.groupby('os_name')['click'].apply(lambda x: x.sum()),
        'b':train.groupby('os_name')['click'].apply(lambda x: x.count()-x.sum())}))[:3]
os_name_test = train.groupby('os_name').nunique()
print(os_name_test.count()[os_name_test.max()==1].index)#查看与os_name唯一对应的特征，os

#%%
#osv 操作系统版本
print(len(train['osv'].unique()))#301个
train['osv'].unique()
osv_look = train['osv'].value_counts().reset_index()
osv_test = train.groupby('osv').nunique()
print(osv_test.count()[osv_test.max()==1].index)#查看与osv唯一对应的特征，无
train.groupby('osv')['click'].apply(lambda x: x.sum()/x.count())
#数据需要清洗整理统一起来，太多乱七八糟的了
#%%
#os 操作系统
print(len(train['os'].unique()))#3个
train['os'].unique()
train['os'].value_counts()#第三类只有18个，考虑融合到另外一个里
os_test = train.groupby('os').nunique()
print(os_test.count()[os_test.max()==1].index)#查看与osv唯一对应的特征，无
train.groupby('os')['click'].apply(lambda x: x.sum()/x.count())
#与os_name完全一致，考虑去掉一个
#%%
#数据预处理
#数据噪音较多，所以打算通过预处理使得模型更具泛化性，同时挖掘更多特征。
#%%
#提取广告投放时间信息，日期、小时以及早中晚时间段
data['hour'] = data['time'].apply(lambda x:int(time.strftime("%H", time.localtime(x))))
data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))

def getSeg(x):
    if x >=0 and x<= 6:
        return 1
    elif x>=7 and x<=12:
        return 2
    elif x>=13 and x<=18:
        return 3
    elif x>=19 and x<=23:
        return 4
data['hour_seg'] = data['hour'].apply(lambda x: getSeg(x))

hourDF = data.groupby(['hour_seg', 'click'])['hour'].count().unstack('click').fillna(0)
hourDF[[0,1]].plot(kind='bar', stacked=True)
#%%
#对操作系统及其版本、名称进行处理
data['os'].replace(0, 1, inplace=True)#操作系统
#版本脏数据统一，清洗
lst = []
for va in data['osv'].values:
    va = str(va)
    va = va.replace('iOS', '')#去掉系统名，用数字代替唯一性方便统一
    va = va.replace('android', '')
    va = va.replace(' ', '')#去空格
    va = va.replace('iPhoneOS', '')
    va = va.replace('_', '.')#统一符号
    va = va.replace('Android5.1', '.')
    try:
        int(va)#int()无法处理字符串化的浮点型
        lst.append(np.nan)
    except:
        sp = ['nan', '11.39999961853027', '10.30000019073486', 'unknown', '11.30000019073486']
        if va in sp:
            lst.append(np.nan)
        elif va == '3.0.4-RS-20160720.1914':
            lst.append('3.0.4')
        else:
            lst.append(va)
temp = pd.Series(lst).value_counts()
temp = temp[temp <= 2].index.tolist()#数量太少的直接缺失值处理
for i in range(len(lst)):
    if lst[i] in temp:
        lst[i] = np.nan
data['osv'] = lst
#处理5.0和5.0.0的统一问题
lst1 = []
for va in data['osv'].values:
    va = str(va).split('.')
    if len(va) < 3:
        va.extend(['0', '0', '0'])
    lst1.append(str(va[0])+'.'+str(va[1])+'.'+va[2])
data['osv'] = lst1

#%%
#品牌的数据清洗
make_look = data['make'].value_counts().reset_index()
lst = []
for va in data['make'].values:
    va = str(va)
    if ',' in va:
        lst.append(va.split(',')[0].lower())#iPhone8,4
    elif ' ' in va:
        lst.append(va.split(' ')[0].lower())#分词取前面的
    elif '-' in va:
        lst.append(va.split('-')[0].lower())
    else:
        lst.append(va.lower())
for i in range(len(lst)):
    if 'iphone' in lst[i]:
        lst[i] = 'apple'
    elif 'redmi' in lst[i]:
        lst[i] = 'xiaomi'
    elif 'vivo' in lst[i]:
        lst[i] = 'vivo'
    elif 'oppo' in lst[i]:
        lst[i] = 'oppo'
    elif lst[i]=='mi':
        lst[i] = 'xiaomi'
    elif lst[i]=='meitu':
        lst[i] = 'meizu'
    elif lst[i]=='nan':
        lst[i] = np.nan
    elif lst[i]=='honor':
        lst[i] = 'huawei'
    elif lst[i]=='le' or lst[i]=='letv' or lst[i]=='lemobile' or lst[i]=='lephone' or lst[i]=='blephone':
        lst[i] = 'leshi'
data['make'] = lst

#%%
#model清洗
model_look = data['model'].value_counts().reset_index()
lst = []
for va in data['model'].values:#统一机型格式
    va = str(va)
    if ',' in va:
        lst.append(va.replace(',',' '))
    elif '+' in va:
        lst.append(va.replace('+',' '))
    elif '-' in va:
        lst.append(va.replace('-',' '))
    elif 'nan'==va:
        lst.append(np.nan)
    else:
        lst.append(va)
data['model'] = lst
#%%
# fillna缺失值填充，用-1代表新类
#缺失特征：user_tags，make，model，osv，app_cate_id，f_channel，app_id
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))
#%%
# replace，把布尔型特征更改
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink',
           'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])

#%%
# 增加交叉特征，把类别型交叉生成粒度更细的分类特征
first_feature = ['app_cate_id', 'app_id']#inner_slot_id粒度过细
second_feature = ["make", "model", "osv", "adid", "advert_name", "campaign_id", "creative_id",
                  "carrier", "nnt", "devtype", "os"]#'campaign_id', 'creative_id'覆盖了基本创意类型因为一个活动id唯一对应是最细的尺度，订单id过细
cross_feature = []
for feat_1 in first_feature:
    for feat_2 in second_feature:
        col_name = "cross_" + feat_1 + "_and_" + feat_2
        cross_feature.append(col_name)
        data[col_name] = data[feat_1].astype(str).values + '_' + data[feat_2].astype(str).values
#将媒体特征与广告和上下文特征交叉形成更细的粒度
#%%
# labelencoder，将分类变量编码
encoder = ['city', 'province', 'make', 'model', 'osv', 'os', 'adid', 'advert_id', 'orderid',
           'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
           'app_id', 'inner_slot_id', 'advert_name', 'osv'
           ]#os和os_name取一个，creative_is_js，creative_is_voicead，app_paid值唯一剔除
encoder = encoder + cross_feature
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

#%%
#记数特征创建
def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    return data
#
for i in cross_feature:
    n = data[i].nunique()
    if n > 5:
        print(i)
        data = feature_count(data, [i])#构造交叉特征对应的记数特征
    else:
        print(i, ':', n)
#%%
# user_tags CountVectorizer
train_new = pd.DataFrame()
test_new = pd.DataFrame()
train = data[:train.shape[0]]
test = data[train.shape[0]:]
train_y = train['click']

cntv = CountVectorizer()
cntv.fit(train['user_tags'])
train_a = cntv.transform(train['user_tags'])
test_a = cntv.transform(test['user_tags'])
train_new = sparse.hstack((train_new, train_a), 'csr')#hstack ： 将矩阵按照列进行拼接，对应的列数必须相等，hstack(blocks, format=None, dtype=None)
test_new = sparse.hstack((test_new, test_a), 'csr')
SKB = SelectPercentile(chi2, percentile=95).fit(train_new, train_y)#区别：SelectKBest选择排名排在前n个的变量 SelectPercentile 选择排名排在前n%的变量 
train_new = SKB.transform(train_new)
test_new = SKB.transform(test_new)
'''
在稀疏矩阵存储格式中：
# - COO 格式在构建矩阵时比较高效
# - CSC 和 CSR 格式在乘法计算时比较高效
A.todense()
# 可以转化为普通矩阵：
'''
#%%
#adid统计特征，不同种类数量（已经创建了记录数的统计，现在是一个特征对应另外一个特征的种类）
## 由于adid是次样本层级的粒度，是聚集到点击率的层面所以是重要的特征，adid基本与广告信息表一一对应，我们象征性的选择广告id与其他挑选出来的id进行特征nunique统计
adid_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'app_id', 'carrier', 'nnt', 'devtype',
            'app_cate_id', 'inner_slot_id']
for feat in adid_nuq:
    gp1 = data.groupby('adid')[feat].nunique().reset_index().rename(columns={feat: "adid_%s_nuq_num" % feat})
    gp2 = data.groupby(feat)['adid'].nunique().reset_index().rename(columns={'adid': "%s_adid_nuq_num" % feat})
    data = pd.merge(data, gp1, how='left', on=['adid'])
    data = pd.merge(data, gp2, how='left', on=[feat])
#%%
## 广告主#38个广告主，是次广告id层级的粒度，广告主与其他特征没有一一对应，所以具备挖掘出其特征的价值，来构造特征更好的表达
#虽然订单id936的粒度并不算粗，大行业的24是太粗，然而由于订单对于是否点击可以看作由底下的活动等因素构成，所以不考虑构造特征
advert_id_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'app_id', 'carrier', 'nnt','devtype',
                 'app_cate_id', 'inner_slot_id']
for fea in advert_id_nuq:
    gp1 = data.groupby('advert_id')[fea].nunique().reset_index().rename(columns={fea: "advert_id_%s_nuq_num" % fea})
    gp2 = data.groupby(fea)['advert_id'].nunique().reset_index().rename(
        columns={'advert_id': "%s_advert_id_nuq_num" % fea})
    data = pd.merge(data, gp1, how='left', on=['advert_id'])
    data = pd.merge(data, gp2, how='left', on=[fea])
#%%
## app_id 
#439个app_id，而app分类只有23个粒度太细，一个app_id唯一对应一个app分类，与其他特征无次关系，值得挖掘其种类数量特征
app_id_nuq = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'f_channel', 'carrier', 'nnt', 'devtype',
              'app_cate_id', 'inner_slot_id']
for fea in app_id_nuq:
    gp1 = data.groupby('app_id')[fea].nunique().reset_index().rename(columns={fea: "app_id_%s_nuq_num" % fea})
    gp2 = data.groupby(fea)['app_id'].nunique().reset_index().rename(columns={'app_id': "%s_app_id_nuq_num" % fea})
    data = pd.merge(data, gp1, how='left', on=['app_id'])
    data = pd.merge(data, gp2, how='left', on=[fea])
#%%
## user_id 搜集用户信息进行构建，一个订单id对应用户数这样的特征是很有意义的，这样的特征挖掘确实高明，到时候把粒度过细会过拟合的特征删除就行了
user_id = ['model', 'make', 'os', 'city', 'province', 'user_tags', 'campaign_id']
data['user_id'] = data['model'].astype(str) + data['make'].astype(str) + data['city'].astype(str) + data[
    'province'].astype(str) + data['user_tags'].astype(str)
gp1 = data.groupby('adid')['user_id'].nunique().reset_index().rename(columns={'user_id': "adid_user_id_nuq_num"})
gp2 = data.groupby('user_id')['adid'].nunique().reset_index().rename(columns={'adid': "user_id_adid_nuq_num"})
data = pd.merge(data, gp1, how='left', on=['adid'])
data = pd.merge(data, gp2, how='left', on=['user_id'])
del data['user_id']

#%%
# add ctr feature
data['period'] = data['day']
data['period'][data['period'] < 27] = data['period'][data['period'] < 27] + 31
for feat_1 in ['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    res = pd.DataFrame()
    temp = data[[feat_1, 'period', 'click']]
    for period in range(27, 35):
        if period == 27:
            count = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] <= period).values].count()).reset_index(name=feat_1 + '_all')
            count1 = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] <= period).values].sum()).reset_index(name=feat_1 + '_1')
        else:
            count = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] < period).values].count()).reset_index(name=feat_1 + '_all')
            count1 = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['period'] < period).values].sum()).reset_index(name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        count.fillna(value=0, inplace=True)
        count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / count[feat_1 + '_all'], 5)
        count['period'] = period
        count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
        count.fillna(value=0, inplace=True)
        res = res.append(count, ignore_index=True)
    print(feat_1, ' over')
    data = pd.merge(data, res, how='left', on=[feat_1, 'period'])
#%%
# 减小内存
for i in data.columns:
    if (i != 'instance_id'):
        if (data[i].dtypes == 'int64'):
            data[i] = data[i].astype('int16')
        if (data[i].dtypes == 'int32'):
            data[i] = data[i].astype('int16')

drop = ['click', 'time', 'f_channel','os_name', 'user_tags',
        'app_paid', 'creative_is_js', 'creative_is_voicead']
#%%
train = data[:train.shape[0]]
test = data[train.shape[0]:]
#del data
gc.collect()
y_train = train.loc[:, 'click']

test = pd.read_table(f'{path}/round1_iflyad_test_feature.txt')
res = test.loc[:, ['instance_id']]

train.drop(drop, axis=1, inplace=True)
print('train:', train.shape)
test.drop(drop, axis=1, inplace=True)
print('test:', test.shape)

X_loc_train = train.values
y_loc_train = y_train.values
X_loc_test = test.values
del train
del test
gc.collect()

# hstack CountVectorizer
X_loc_train = sparse.hstack((X_loc_train, train_new), 'csr')
X_loc_test = sparse.hstack((X_loc_test, test_new), 'csr')
del train_new
del test_new
gc.collect()

# 利用不同折数加参数，特征，样本（随机数种子）扰动，再加权平均得到最终成绩
lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.02, n_estimators=6000,
                             max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                             min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_alpha=3, reg_lambda=0.1, seed=1000, n_jobs=-1, silent=True)
skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1024).split(X_loc_train,y_loc_train))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    lgb_model = lgb_clf.fit(X_loc_train[train_index], y_loc_train[train_index],
                            eval_names=['train', 'valid'],
                            eval_metric='logloss',
                            eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                      (X_loc_train[test_index], y_loc_train[test_index])], early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']
    test_pred = lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss / 5)

res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score'] / 5

mean = res['predicted_score'].mean()
print('mean:', mean)
res[['instance_id', 'predicted_score']].to_csv(f'{path}/result1.csv', index=False)