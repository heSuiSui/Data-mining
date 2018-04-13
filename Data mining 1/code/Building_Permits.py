from data_analysis import *

csv_path='./Building_Permits.csv'
dataFrame=read_csv(csv_path,None)  

#标称属性
name_category = ['Permit Type', 'Block', 'Lot', 'Street Number', 'Street Number Suffix', 'Street Name', 'Street Suffix',
                 'Current Status', 'Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit',
                 'Existing Use', 'Proposed Use', 'Plansets', 'TIDF Compliance', 'Existing Construction Type', 
                 'Proposed Construction Type', 'Site Permit', 'Supervisor District', 'Neighborhoods - Analysis Boundaries']
#数值属性
name_value = ['Number of Existing Stories', 'Number of Proposed Stories', 'Estimated Cost', 'Revised Cost', 'Existing Units', 'Proposed Units']

#标称属性统计频数
count(dataFrame,name_category)

#数值属性统计最小、最大、均值、中位数、四分位数及缺失值个数
describe(dataFrame,name_value)

#绘制直方图
histogram(dataFrame,name_value)

#绘制qq图
qqplot(dataFrame,name_value)

#绘制盒图
boxplot(dataFrame,name_value)

#将缺失值剔除
df_dropna = dataFrame.dropna()
print(df_dropna.shape)

#通过属性的相关关系来填补缺失值
cols = ['Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit', 'TIDF Compliance']
df_fillna = dataFrame[cols].fillna('N')

print('                  旧数据\n')
count(dataFrame, cols)

print('\n\n                  新数据\n')
count(df_fillna, cols)