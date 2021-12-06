from math import factorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.api import OLS
import statsmodels.api as sm
import seaborn as sns
import sympy as sy


questionNum = 0
def printQuestion():
    global questionNum
    questionNum += 1
    print("\nQuestion #{}".format(questionNum))

# Question 1:
printQuestion()
print("-- No Console Output --")
d1 = pd.read_csv(open("final.csv","r"))

# Question 2:
printQuestion()
print(d1.shape)
print(d1.columns)

# Question 3:
printQuestion()
print(d1.info())

# Question 4:
printQuestion()
del d1['ID'] # or -> d1 = d1.drop(["ID"], axis=1)
print(d1)

# Question 5:
printQuestion()
print("MOFB missing: {}".format(d1['MOFB'].isnull().sum())) # number of missing values for each variable
print("YOB missing: {}".format(d1['YOB'].isnull().sum())) # number of missing values for each variable
print("AOR missing: {}".format(d1['AOR'].isnull().sum())) # number of missing values for each variable
# print(d1.describe())

# Question 6:
printQuestion()
d2 = d1[["RMOB", "WI", "RCA", "Religion", "Region", "AOR", "HEL", "DOBCMC", "DOFBCMC", "MTFBI", "RW", "RH", "RBMI"]]
print(d2)

# Question 7:
printQuestion()
d3 = d2.dropna() # deleting rows that has missing values (deleting all missings)
print(d3)

# Question 8:
printQuestion()
print(d3.describe().transpose())

# Question 9:
printQuestion()
d3["AVG"] = d3[["DOBCMC","DOFBCMC","MTFBI"]].mean(axis=1)
print(d3)

# Question 10:
printQuestion()
d3["Newreligion"] = d3[["Religion"]]
d3["Newreligion"].loc[d3.Newreligion != 1] = 2
print(d3)

# Question 11:
printQuestion()
print(pd.crosstab(index=d3.Region,columns=["Region","Count"]))

# Question 12:
printQuestion()
print(d3.melt(id_vars="Region", value_vars=["Religion"])
         .groupby([pd.Grouper(key='Region'),'value'])
         .size()
         .unstack(fill_value=0))

# Question 13:
printQuestion()
print(pd.crosstab(index=d3["Region"], columns=d3["AOR"]).mean(axis=1))

# Question 14:
printQuestion()
print(pd.crosstab(index=d3["Religion"], columns=d3["AOR"]).var(axis=1))

# --- Needed for #19
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

# Question 15:
printQuestion()
print("-- No Console Output --")
plt.subplot(2,2,1)
plt.boxplot(d3["MTFBI"])

# Question 16:
printQuestion()
print("-- No Console Output --")
plt.subplot(2,2,2)
plt.hist(d3["RCA"])

# Question 17:
printQuestion()
print("-- No Console Output --")
regionCrosstab = pd.crosstab(index=d3["Region"],columns="Count")
regionCrosstab.plot.bar(rot=360,title="Total for Regions", layout=(2,2), ax=ax3)

# Question 18:
printQuestion()
print("-- No Console Output --")
pd.crosstab(index=d3["Region"],columns="Count").plot.pie(subplots=True,title="Pie chart of Regions", ax=ax4)
# plt.show()

# Question 19:
printQuestion()
print("-- No Console Output --")
plt.suptitle("Questions 15-18")

# Question 20:
printQuestion()
print("-- No Console Output --")
d4 = d3.groupby(d3['WI'])

# Question 21:
printQuestion()
for key in d4.groups:
    group = d4.get_group(key)
    print("WI[\"{}\"]".format(key))
    print("Mean: {} | Min: {} | Max: {} | STD: {} | VAR: {}".format(
        group.mean()["MTFBI"],
        group.min()["MTFBI"],
        group.max()["MTFBI"],
        group.std()["MTFBI"],
        group.var()["MTFBI"]))
    print()
    

# Question 22:
printQuestion()
print(stats.ttest_1samp(d3.MTFBI, 30))

# Question 23:
printQuestion()
print(stats.shapiro(d3.MTFBI))

# Question 24:
printQuestion()
print(stats.ttest_ind(d3[d3.Newreligion==1].MTFBI,d3[d3.Newreligion==2].MTFBI))

# Question 25:
printQuestion()
correlation= d3[["DOBCMC", "DOFBCMC", "AOR", "MTFBI", "RW", "RH","RBMI"]].corr()
plt.matshow(correlation)
print(correlation)

# Question 26:
printQuestion()
print("-- No Question --")

# Question 27:
printQuestion()
y=d3.MTFBI
x=sm.add_constant(d3[["AOR", "RW", "Region"]])
model = OLS(y, x).fit()
print(model.summary())

# --- Needed for 28-32

trueMean = 640
trueVariance = 257920
def sim(n):
    X=np.random.binomial(20,.7,n)
    U=np.random.uniform(15,30,n)
    N=np.random.normal(0,5,n)
    E=np.random.uniform(-1,-1,n)

    y= 50 + 10*X + 20*U + 100*N + E
    y1=pd.DataFrame(y)
    return y1

# Question #28:
printQuestion()
simulation = sim(100)
print(simulation)

# Question #29:
printQuestion()
firstTest = pd.concat([sim(1000) for i in range(100)])
testMean= np.mean(firstTest)
print(abs(testMean-trueMean))

# Question #30:
printQuestion()
testVariance = np.var(firstTest)
print(abs(testVariance-trueVariance))

# Question #31:
printQuestion()
secondTest=pd.concat([sim(1000) for i in range(500)])
secondTestMean=np.mean(secondTest)
print(abs(secondTestMean-trueMean))

# Question #32:
printQuestion()
secondTestVariance=np.var(secondTest)
print(abs(secondTestVariance-trueVariance))

# Question #33:
printQuestion()
for x in range(1,6):
    y = x + 1
    z = x + 2
    dividend = (np.e ** x)-np.log(z**z)
    divisor = (5+y)
    result = dividend / divisor
    print(result)

# Question #34:
printQuestion()
a = np.array([[70,100,40],[120,450,340],[230,230,1230]])
b = np.array([900,1000,3000])
print(np.linalg.solve(a,b))

# Question #35:
printQuestion()
A = np.array([[20,30,30],[20,80,120],[40,90,360]])
print(np.linalg.inv(A))

# Question #36:
printQuestion()
b = np.array([10,20,30])
result = np.invert(np.matrix.transpose(A) * A) * np.matrix.transpose(A) * b
print(result)

# Question #37:
printQuestion()
x = range(2,16)
y = []
for num in x:
    y.append((np.e**num)/(factorial(num)))

plt.figure()
plt.plot(x,y)

# Question #38:
printQuestion()
x = np.linspace(-1000, 1000)
y = []
for num in x:
    if (num < 0):
        y.append((2*(num**2)) + np.e**num + 3)
    elif (num >= 0 or num < 10):
        y.append((9*num) + np.log(20))
    else:
        y.append((7*(num**2)) + (5*num) - 17)

fig,ax= plt.subplots()
ax = sns.lineplot(x=x, y=y)


# Question #39:
printQuestion()
for x in range(10,20):
    print("Circle radius {} with area {}".format(x,np.pi * x**2))

# Question #40:
printQuestion()
result = 0
for x in range(2, 10001):
    result += (1/np.log(x))
print(result)

# Question #41:
printQuestion()
result = 0
for i in range(1, 31):
    for j in range(1, 11):
        result += (i**10) / (3+j)
print(result)

# Question #42:
printQuestion()
x = sy.Symbol("x")
result = sy.integrate(((x**15) * (np.e**(-40 * x))), (x,0,np.Infinity))
print(result)

# Question #43:
printQuestion()
x = sy.Symbol("x")
result = sy.integrate(((x**150) * (1 - 30)**30), (x,0,1))
print(result)

# Question #44:
printQuestion()
print("duplicate question of #33")
for x in range(1,6):
    y = x + 1
    z = x + 2
    dividend = (np.e ** x)-np.log(z**z)
    divisor = (5+y)
    result = dividend / divisor
    print(result)

# Question #45:
printQuestion()
x = sy.Symbol("x")
print(sy.solve((x**2) - (33 * x) + 1), x)

# Question #46:
printQuestion()
print("--Informed to not do #46--")

# Question #47:
printQuestion()
p = 40
t = 50
r = 0.10
print(p * (1 + r)**t)

# Question #48:
printQuestion()
model = sm.OLS(d3.MTFBI, d3.AOR).fit()
model.predict(d3.AOR)
print(model.summary())

# Question #49:
printQuestion()
print(stats.pearsonr(d3.AOR, d3.MTFBI))

# ---- Function given in project document:
def chi_sq_test_for_variance(variable,h0):  #function for the chi squared distribution associated with the above formula

    
    sample_variance = variable.var()                 # Find the variance of the sample
    n = variable.notnull().sum()                     # Take the sum of the number of values that are not missing 
                                                     # the actual number of observations for the variable where
                                                     # True = 1, False = 0
    degrees_of_freedom = n-1                         # Find the degrees of freedom
    x_sq_stat = (n - 1) * sample_variance / h0       # Using the formula above to calculate the X^2 statistic
    p = stats.chi2.cdf(x_sq_stat,degrees_of_freedom) # Here, a cumulative distribution function is used to determine
                                                     # the significance of the variance using the X^2 statistic.
        
                                                     # If a chi square test statistic is over the 99th percentile, 
                                                     # we'd have reason to suspect significance at alpha = .05.
                                                     # We need to account for circumstance where the p value is greater
                                                     # than .05, however:
    if p > .05:                                                                
        p = 1 - p
    return (x_sq_stat, p,degrees_of_freedom)         # End of function     
    

# Question #50:
printQuestion()
dbp_variance = round(d3["AOR"].var(),2)  
x_sq_stat, pval, dof = chi_sq_test_for_variance(d3["AOR"],h0=10)
print(dbp_variance)


# ---- Show all plots
plt.show()