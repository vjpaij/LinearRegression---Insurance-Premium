import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    #plt.show()


#try_parameters(267, -2091)


#using library for linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#since model.fit needs 2D for inputs, we assign input as below
inputs = non_smoker_df[['age']]
targets = non_smoker_df['charges']
model.fit(inputs, targets)
print(model.predict(np.array([[23], [36], [60]])))
predictions = model.predict(inputs)
print(predictions)
#m value is stored in coef_ and b value in intercept_
print(f"m: {model.coef_}\ny: {model.intercept_}")

#Using SGDRegressor class - Used for more data. (stochastic gradient descent)
from sklearn.linear_model import SGDRegressor 
model_SGD = SGDRegressor()
#since model.fit needs 2D for inputs, we assign input as below
model_SGD.fit(inputs, targets)
predictions_SGD = model_SGD.predict(inputs)
print(predictions_SGD)

#Any difference in Linear Regression and SGD?
print(model.predict(np.array([[23], [36], [60]])))
print(model_SGD.predict(np.array([[23], [36], [60]])))
try_parameters(model.coef_, model.intercept_)
try_parameters(model_SGD.coef_, model_SGD.intercept_)
#Build similar model for different input parameters

#to calculate the loss/error between the predicted and actual value
#root mean square error
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))

loss = rmse(targets, predictions)
loss_SGD = rmse(targets, predictions_SGD)
print(f"Loss of Linear Regression Model: {loss}\nLoss of SGD model: {loss_SGD}")

#Linear Regression using Multiple Features
#charges = (m1 * age) + (m2 * bmi) + (m3 * children) + b
multi_inputs = non_smoker_df[['age', 'bmi', 'children']]
multi_model = LinearRegression()
multi_model.fit(multi_inputs, targets)
multi_predictions = multi_model.predict(multi_inputs)
multi_loss = rmse(targets, multi_predictions)
print("Loss due to Age, BMI & Children Numeric Feature combination: ", multi_loss)

#Using Categorical Features in the model
'''
3 rules used:
1. if only 2 categories (binary), we can replace their values with 0 and 1.
2. more than 2 categories, we perform one-hot encoding
3. if the categories are in a natural order i.e. ordinal, then we can convert them into a number preserving the order
eg: (cold, neutal, warm, hot) can be set as (1, 2, 3, 4)
'''
#Smoker Feature
smoker_codes = {'no': 0, 'yes': 1}
data['smoker_codes'] = data['smoker'].map(smoker_codes)
#Sex Feature
sex_codes = {'male': 0, 'female': 1}
data['sex_codes'] = data['sex'].map(sex_codes)

#model with these 2 categorical features: charges = (m1 * age) + (m2 * bmi) + (m3 * children) + 
#                                                   (m4 * smoker_codes) + (m5 * sex_codes) + b
cat_inputs = data[['age', 'bmi', 'children', 'smoker_codes', 'sex_codes']]
all_targets = data['charges']
cat_model = LinearRegression()
cat_model.fit(cat_inputs, all_targets)
cat_predictions = cat_model.predict(cat_inputs)
cat_loss = rmse(all_targets, cat_predictions)
print("Loss due to Age, BMI, Children, Smoker & Sex combination: ", cat_loss)

#Region Feature
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(data[['region']])
print(enc.categories_)
one_hot = enc.transform(data[['region']]).toarray()
print(one_hot)
data[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

all_inputs = data[['age', 'bmi', 'children', 'smoker_codes', 'sex_codes', 'northeast', 'northwest', 'southeast', 'southwest']]
all_model = LinearRegression()
all_model.fit(all_inputs, all_targets)
all_predictions = all_model.predict(all_inputs)
all_loss = rmse(all_targets, all_predictions)
print("Loss due to all features: ", all_loss)

#Needs improvisation as model wasnt as expected
print(all_model.coef_)
'''
[  256.85635254   339.19345361   475.50054515 23848.53454191
   131.3143594    587.00923503   234.0453356   -448.01281436
  -373.04175627]
  so we see age has a weight of 256 whereas bmi has 339 and children has 475. But we have seen earlier that age has a 
  greater weightage than these 2. So whats happening here? This is due to smaller values for age wrt to other fields 
  and hence these values has to be standardized to a proportionate value before creating a model
'''
from sklearn.preprocessing import StandardScaler
numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(data[numeric_cols])
scaled_inputs = scaler.transform(data[numeric_cols])
cat_cols = ['smoker_codes', 'sex_codes', 'northeast', 'northwest', 'southeast', 'southwest']
cat_data = data[cat_cols].values

new_inputs = np.concatenate((scaled_inputs, cat_data), axis=1)
new_model = LinearRegression()
new_model.fit(new_inputs, all_targets)
new_predictions = new_model.predict(new_inputs)
new_loss = rmse(all_targets, new_predictions)
print("Loss due to all features after Standardization: ", new_loss)
print(new_model.coef_)
'''
[ 3607.47273619  2067.69196584   572.99820995 23848.53454191
   131.3143594    587.00923503   234.0453356   -448.01281436
  -373.04175627]
'''













