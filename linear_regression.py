import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("insurance.csv")
data.info()
data.describe()
print(data.isnull().sum())

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

#Age feature
data['age'].describe()

fig1 = px.histogram(data, x='age', marginal='box', nbins=47, title='Distribution of Age')
fig1.update_layout(bargap = 0.1)
#fig1.show()
'''
The distribution of age is almost uniform i.e. 20-30 customers across all age group except age 18 & 19.
'''

#BMI feature
fig2 = px.histogram(data, x='bmi', marginal='box', color_discrete_sequence=['red'], title='Distribution of BMI')
fig2.update_layout(bargap = 0.1)
#fig2.show()
'''
Distribution of BMI is showing Gaussian/Normal distribution
'''

#Charges feature
'''
This is the column we are trying to predict. We are also using categorical column 'smoker' to distinguish the charges
'''
fig3 = px.histogram(data, x='charges', marginal='box', color= 'smoker', color_discrete_sequence=['green', 'grey'], 
                    title='Annual Medical Charges')
fig3.update_layout(bargap = 0.1)
#fig3.show()
'''
The distribution shows power law
'''

#Smoker feature
data['smoker'].value_counts()
fig4 = px.histogram(data, x='smoker', color= 'sex', color_discrete_sequence=['yellow', 'brown'], title='Smoker')
#fig4.show()

#Age and Charges relationship
fig5 = px.scatter(data, x='age', y='charges', color='smoker', opacity=0.8, hover_data=['sex'], title='Age vs Charges')
fig5.update_traces(marker_size = 5)
#fig5.show()
'''
We see 3 distinguished clusters. With no smokers, a mix of both and then with smokers
'''

#BMI vs Charges relationship
fig6 = px.scatter(data, x='bmi', y='charges', color='smoker', opacity=0.8, hover_data=['sex'], title='BMI vs Charges')
fig6.update_traces(marker_size = 5)
#fig6.show()
'''
it looks for non-smokers, bmi doesnt play much significance in increase in charges but we do see the trend for smokers.
'''

#Correlation
'''
From above relationship we can see some trends. Now we need to establish the correlation
'''
age_charge_corr = data['charges'].corr(data['age'])
print(f"Age - Charge Correlation {age_charge_corr}")
bmi_charge_corr = data['charges'].corr(data['bmi'])
print(f"BMI - Charge Correlation {bmi_charge_corr}")

#Correlation for categorical features.
#Categorical feature should be changed to numeric to find the correlation
smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = data['smoker'].map(smoker_values)
print(smoker_numeric)
smoker_charge_corr = data['charges'].corr(smoker_numeric)
print(f"Smoker - Charge Correlation {smoker_charge_corr}")

#Correlation against each features (numeric columns only)
num_data = data.select_dtypes(include=['number'])
print(num_data.corr())
#representing it in a heatmap
# sns.heatmap(num_data.corr(), cmap='Blues', annot=True)
# plt.title('Correlation Matrix')
#plt.show()

#Linear Regression using a Single Feature
#Smoker and Age features have strongest correlation with charges
#As Smoker is a categorical, building a linear regression is not significant
#So first lets build Linear Regression for Age for non-smokers against the charges
non_smoker_df = data[data['smoker'] == 'no']
fig7 = px.scatter(non_smoker_df, x='age', y='charges', opacity=0.8, title='Age(Non-Smoker) vs Charges')
#fig7.show()
'''
y = mx + b
charges = (m * age) + b
'''
def estimate_charges(age, m, b):
    return (m * age) + b

#lets assume m=50 and b=100
m, b = 50, 100
ages = non_smoker_df['age']
estimated_charges = estimate_charges(ages, m, b)

#now we will check how good the above value was compared to actual data
target = non_smoker_df['charges']
# plt.plot(ages, estimated_charges, 'r', alpha=0.9)
# plt. scatter(ages, target, s = 8, alpha=0.8)
# plt.xlabel('Age')
# plt.ylabel('Charges')
# plt.legend(['Estimate', 'Actual'])
# plt.show()
#above value doesnt fit well with the actual data. Lets build a function to change value of m and b and try to fit.
def try_parameters(m, b):
    ages = non_smoker_df['age']
    target = non_smoker_df['charges']
    estimated_charges = estimate_charges(ages, m, b)
    plt.plot(ages, estimated_charges, 'r', alpha=0.9)
    plt. scatter(ages, target, s = 8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    plt.show()

try_parameters(400, -6500)




