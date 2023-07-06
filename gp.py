#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import operator
import random
import math
from copy import deepcopy
from functools import partial
from deap import gp, base, creator, tools, algorithms
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


PATH_TRAIN = "./data/train.csv"
PATH_TEST = "./data/test.csv"

DTYPES_FEATURES = {
          "id": "uint64",
          "fr_COO": "category",
          "fr_COO2": "category",
      }

DTYPES_TARGETS = {
          "EC1": "bool",
          "EC2": "bool",
          "EC3": "bool",
          "EC4": "bool",
          "EC5": "bool",
          "EC6": "bool"
}

DROP_COLS = ["EC3", "EC4", "EC5", "EC6"]



def _load_data(datapath: str, dtypes: dict, drop_cols: list) -> pd.DataFrame:
  return pd.read_csv(
      filepath_or_buffer=datapath,
      dtype=dtypes,
      index_col="id"
    ).drop(columns=drop_cols, axis=1)


GetTrainDF = partial(_load_data, datapath=PATH_TRAIN, dtypes=dict(**DTYPES_TARGETS, **DTYPES_FEATURES), drop_cols=DROP_COLS)
GetTestDF = partial(_load_data, datapath=PATH_TEST, dtypes=DTYPES_FEATURES, drop_cols=[])

df_test = GetTestDF().astype(float)
df_train = GetTrainDF().astype(float)
df_train.head()


# In[3]:


# names (=argn) and number (=argc) of arguments

argn = df_train.drop(columns=["EC1", "EC2"], inplace=False).columns.to_list()
argc = len(argn)


# In[4]:


pset = gp.PrimitiveSet("MAIN", arity=argc, prefix="ARG")

pset.renameArguments(**{f"ARG{i}": arg for i, arg in enumerate(argn)})

def protectedDiv(left: float, right: float) -> float:
    if right == 0:
        return 1
    else:
         return left / right
    
def if_lt(a: float, b: float, c: float, d: float) -> float:
    if a < b:
        return c
    else:
        return d
    
pset.addPrimitive(if_lt, 4, name="if_lt")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(operator.add, 2, name="add")
pset.addPrimitive(operator.sub, 2, name="sub")
pset.addPrimitive(operator.mul, 2, name="mul")
pset.addPrimitive(operator.neg, 1, name="neg")
pset.addPrimitive(math.cos, 1, name="cos")
pset.addPrimitive(math.sin, 1, name="sin")
pset.addPrimitive(math.tanh, 1, name="tanh")
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))


# In[5]:


# create a fitness and individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# In[6]:


def get_proba(individual: str, X: np.ndarray, pset: gp.PrimitiveSet=pset) -> np.ndarray:
    """Get the probabilities of the positive class for each sample in X"""
    func = gp.compile(expr=individual, pset=pset)
    y_pred_vals = np.array([func(*x) for x in X])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_pred_vals.reshape(-1, 1))
    y_pred_probs = scaler.transform(y_pred_vals.reshape(-1, 1)).reshape(-1)

    return y_pred_probs

def get_pred(individual: str, X: np.ndarray, pset: gp.PrimitiveSet=pset) -> np.ndarray:
    """Get the predicted class for each sample in X"""
    y_pred_probs = get_proba(individual, X, pset=pset)
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)
    
    return y_pred

def evalBinaryClassification(individual: str, X: np.ndarray, y: np.ndarray, pset: gp.PrimitiveSet=pset):
    """fitness function that takes an individual, X and y as input and returns the corresponding auc score"""
    y_pred_probs = get_proba(individual, X, pset=pset)
    return roc_auc_score(y, y_pred_probs),


# In[7]:


# create a toolbox

toolbox_ec1 = base.Toolbox()
toolbox_ec2 = base.Toolbox()


# In[8]:


# create a X and y

X_ec1 = df_train.drop(columns=["EC1", "EC2"], inplace=False).to_numpy()
X_ec2 = deepcopy(X_ec1)
y_ec1 = df_train["EC1"].to_numpy().astype(int)
y_ec2 = df_train["EC2"].to_numpy().astype(int)


# # Optional improvements:
# 
# ## 1. Balancing the Dataset

# In[9]:


# the dataset is unbalanced
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
sns.countplot(x="EC1", data=df_train, ax=ax[0])
sns.countplot(x="EC2", data=df_train, ax=ax[1])
ax[0].set_title("EC1 - Original dataset")
ax[1].set_title("EC2 - Original dataset")


X = df_train.drop(["EC1", "EC2"], axis=1).to_numpy()
y_ec1 = df_train["EC1"].to_numpy()
y_ec2 = df_train["EC2"].to_numpy()


print(f"Number of Cases before Oversampling: {len(y_ec1)}")

# we will use SMOTE/ADASYN/RANDOMSAMPLER to oversample the minority class inside the df_train dataframe
# oversample the minority class (uncomment the sampler you want to use)

samplerMod = SMOTE(random_state=42)                 # -> synthetic minority oversampling technique
# samplerMod = ADASYN(random_state=42)              # -> adaptive synthetic sampling approach
# samplerMod = RandomOverSampler(random_state=42)   # -> random oversampling


X_ec1, y_ec1 = samplerMod.fit_resample(X, y_ec1)
X_ec2, y_ec2 = samplerMod.fit_resample(X, y_ec2)

# check if dataset is balanced
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
sns.countplot(x=y_ec1, ax=ax[0])
sns.countplot(x=y_ec2, ax=ax[1])

# set the title of the plot
ax[0].set_title("EC1 - Oversamplet dataset")
ax[1].set_title("EC2 - Oversamplet dataset")

plt.show()

print(f"Number of Cases after Oversampling: {len(y_ec1)}")


# ## 2. Min/Max Scaling of the features

# In[10]:


# lets use a MinMaxScaler to scale the data

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_ec1)
X_ec1 = scaler.transform(X_ec1)
X_ec2 = scaler.transform(X_ec2)


# In[11]:


# set up the toolbox for the gp algorithm on the ec1 target

toolbox_ec1.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox_ec1.register("individual", tools.initIterate, creator.Individual, toolbox_ec1.expr)
toolbox_ec1.register("population", tools.initRepeat, list, toolbox_ec1.individual)


toolbox_ec1.register("select", tools.selTournament, tournsize=2)
toolbox_ec1.register("mate", gp.cxOnePoint)
toolbox_ec1.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_ec1.register("mutate", gp.mutUniform, expr=toolbox_ec1.expr_mut, pset=pset)
toolbox_ec1.register("compile", gp.compile, pset=pset)
toolbox_ec1.register("evaluate", evalBinaryClassification, X=X_ec1, y=y_ec1)

# bloat control
toolbox_ec1.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox_ec1.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)

# set up the toolbox for the gp algorithm on the ec2 target

toolbox_ec2.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox_ec2.register("individual", tools.initIterate, creator.Individual, toolbox_ec2.expr)
toolbox_ec2.register("population", tools.initRepeat, list, toolbox_ec2.individual)


toolbox_ec2.register("select", tools.selTournament, tournsize=2)
toolbox_ec2.register("mate", gp.cxOnePoint)
toolbox_ec2.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_ec2.register("mutate", gp.mutUniform, expr=toolbox_ec2.expr_mut, pset=pset)
toolbox_ec2.register("compile", gp.compile, pset=pset)
toolbox_ec2.register("evaluate", evalBinaryClassification, X=X_ec2, y=y_ec2)

toolbox_ec2.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox_ec2.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)


# In[12]:


# statistics dictionary
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# In[13]:


# run the algorithm on the ec1 target

# evolutionary parameters
NGEN_EC1 = 100
POPSIZE_EC1 = 1200
CXPB_EC1 = 0.9
MUTPB_EC1 = 0.1

pop_ec1 = toolbox_ec1.population(n=POPSIZE_EC1)
hof_ec1 = tools.HallOfFame(1)


pop_ec1, log_ec1 = algorithms.eaSimple(pop_ec1, toolbox_ec1, CXPB_EC1, MUTPB_EC1, NGEN_EC1, stats=stats, halloffame=hof_ec1, verbose=True)

# save the winner program as a string in lisp format
try:
    f_ec1_raw = hof_ec1[0]
    print(f_ec1_raw)
except IndexError:
    print("No program found for EC1")
    f_ec1_raw = None


# In[ ]:


# plot the evolution from the logbook log_ec1
# plot the average, max and std deviation of fitness

try:
    gen_ec1 = log_ec1.select("gen") 
    fit_avgs_ec1 = log_ec1.select("avg")
    fit_maxs_ec1 = log_ec1.select("max")

    fig, ax1 = plt.subplots()
    line2 = ax1.plot(gen_ec1, fit_avgs_ec1, "r-", label="Average")
    line3 = ax1.plot(gen_ec1, fit_maxs_ec1, "g-", label="Maximum")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("ROC AUC Score")

    ax1.set_title("EC1 - Evolution of the fitness over generations")

    ax1.legend()


    plt.show()
except NameError:
    print("No logbook found for EC1")


# In[ ]:


# run the algorithm on the ec2 target

# evolutionary parameters
NGEN_EC2 = 100
POPSIZE_EC2 = 1200
CXPB_EC2 = 0.9
MUTPB_EC2 = 0.1

pop_ec2 = toolbox_ec2.population(n=POPSIZE_EC2)
hof_ec2 = tools.HallOfFame(1)


pop_ec2, log_ec2 = algorithms.eaSimple(pop_ec2, toolbox_ec2, CXPB_EC2, MUTPB_EC2, NGEN_EC2, stats=stats, halloffame=hof_ec2, verbose=True)

# save the winner program as a "string in lisp format
try:
    f_ec2_raw = hof_ec2[0]
    print(f_ec2_raw)
except IndexError:
    print("No program found for EC2")
    f_ec2_raw = None


# In[ ]:


# plot the evolution from the logbook log_ec1
# plot the average, max and std deviation of fitness

try:
    gen_ec2 = log_ec2.select("gen") 
    fit_avgs_ec2 = log_ec2.select("avg")
    fit_maxs_ec2 = log_ec2.select("max")

    fig, ax1 = plt.subplots()
    line2 = ax1.plot(gen_ec2, fit_avgs_ec2, "r-", label="Average")
    line3 = ax1.plot(gen_ec2, fit_maxs_ec2, "g-", label="Maximum")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("ROC AUC Score")

    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax1.set_title("EC2 - Evolution of the fitness over generations")
    ax1.legend()

    plt.show()
except NameError:
    print("No logbook found for EC2")



# In[ ]:


# predict the labels for the test set

X = df_test.to_numpy()

# if we previously scaled the data, we need to scale the test set in the same way
try:
    scaler.fit(X)
    X = scaler.transform(X)
except NameError:
    print("No Scaler found")

# ec1
if f_ec1_raw is not None:
    ec1_pred = get_proba(f_ec1_raw, X, pset=pset)

# ec2
if f_ec2_raw is not None:
    ec2_pred = get_proba(f_ec2_raw, X, pset=pset)

# save the predictions in a csv file
try:
    df_result = pd.DataFrame({"id": df_test.index.values, "EC1": ec1_pred, "EC2": ec2_pred})
    df_result.to_csv("results.csv", index=False)

except NameError:
    print("No predictions found")


# # EC1 - Hall of Fame
# 
# ## 1. Basic GP
# 
# ```{python}
# mul(sub(mul(if_lt(neg(add(div(ExactMolWt, Chi2v), sub(tanh(if_lt(div(if_lt(div(HallKierAlpha, fr_COO2), mul(ExactMolWt, neg(FpDensityMorgan1)), NumHeteroatoms, if_lt(div(SlogP_VSA3, BertzCT), sub(ExactMolWt, Chi1), sub(NumHeteroatoms, FpDensityMorgan1), neg(NumHeteroatoms))), BertzCT), sub(ExactMolWt, tanh(mul(BertzCT, cos(neg(HallKierAlpha))))), sub(if_lt(tanh(Kappa3), neg(add(Chi1, if_lt(sub(tanh(NumHeteroatoms), cos(add(NumHeteroatoms, PEOE_VSA8))), EState_VSA1, sub(div(div(ExactMolWt, EState_VSA1), Chi2v), Kappa3), cos(Chi2n)))), MinEStateIndex, ExactMolWt), Chi1n), div(SlogP_VSA3, cos(Chi2v)))), SMR_VSA10))), div(SlogP_VSA3, div(MinEStateIndex, sin(MinEStateIndex))), VSA_EState9, tanh(Kappa3)), MinEStateIndex), if_lt(if_lt(PEOE_VSA14, Chi4n, sin(PEOE_VSA14), Chi1), neg(add(div(div(ExactMolWt, Chi2v), Chi1), sub(Chi4n, mul(if_lt(sin(Chi2n), neg(PEOE_VSA8), HallKierAlpha, PEOE_VSA14), Chi1v)))), cos(EState_VSA1), PEOE_VSA10)), if_lt(div(HallKierAlpha, neg(add(div(FpDensityMorgan1, Chi2v), sub(Chi4n, if_lt(Chi2v, sub(if_lt(tanh(tanh(Kappa3)), add(div(ExactMolWt, Chi2v), if_lt(sub(tanh(SMR_VSA10), sin(EState_VSA2)), sub(if_lt(tanh(Kappa3), Chi3v, Chi1, FpDensityMorgan1), EState_VSA1), Chi2v, cos(Chi2n))), Chi1, ExactMolWt), mul(BertzCT, MinEStateIndex)), sub(Chi1, Kappa3), mul(if_lt(HallKierAlpha, VSA_EState9, Chi2n, EState_VSA1), neg(FpDensityMorgan1))))))), mul(mul(mul(HallKierAlpha, fr_COO2), neg(PEOE_VSA14)), neg(FpDensityMorgan1)), NumHeteroatoms, if_lt(div(SlogP_VSA3, BertzCT), sub(ExactMolWt, Chi1), if_lt(div(HallKierAlpha, neg(add(div(FpDensityMorgan1, neg(add(div(ExactMolWt, Chi2v), sub(tanh(if_lt(sin(MinEStateIndex), sub(Chi2v, div(SlogP_VSA3, BertzCT)), sub(ExactMolWt, Chi1n), div(mul(BertzCT, MinEStateIndex), sub(NumHeteroatoms, div(Chi1v, MinEStateIndex))))), SMR_VSA10)))), sub(Chi4n, if_lt(Chi2v, sub(if_lt(tanh(Kappa3), add(div(ExactMolWt, Chi2v), if_lt(sub(tanh(SMR_VSA10), sin(EState_VSA2)), tanh(SlogP_VSA3), Chi2v, cos(Chi2n))), Chi1, ExactMolWt), EState_VSA1), sub(Chi1, Kappa3), mul(if_lt(HallKierAlpha, VSA_EState9, Chi2n, EState_VSA1), neg(FpDensityMorgan1))))))), mul(ExactMolWt, neg(FpDensityMorgan1)), NumHeteroatoms, if_lt(div(SlogP_VSA3, BertzCT), sub(add(div(ExactMolWt, Chi2v), sub(tanh(if_lt(div(if_lt(div(HallKierAlpha, EState_VSA1), mul(NumHeteroatoms, neg(FpDensityMorgan1)), NumHeteroatoms, Chi4n), BertzCT), sub(Chi2v, tanh(mul(BertzCT, cos(neg(HallKierAlpha))))), sub(ExactMolWt, Kappa3), div(SlogP_VSA3, cos(FpDensityMorgan3)))), SMR_VSA10)), Chi1), sub(NumHeteroatoms, FpDensityMorgan1), SMR_VSA10)), SMR_VSA10)))
# ```
# 
# *ROC_AUC_SCORE=0.68849*
# 
# ## 2. High Popsize (=1200) and number of Generations (=100)
# 
# 
# ```{python}
# if_lt(if_lt(NumHeteroatoms, if_lt(sub(if_lt(cos(neg(SMR_VSA10)), if_lt(-1, FpDensityMorgan2, sub(PEOE_VSA6, PEOE_VSA6), VSA_EState9), mul(NumHeteroatoms, if_lt(cos(MaxAbsEStateIndex), if_lt(mul(sub(PEOE_VSA8, PEOE_VSA6), HallKierAlpha), add(FpDensityMorgan1, if_lt(Chi3v, sub(if_lt(FpDensityMorgan1, sub(PEOE_VSA6, if_lt(sub(tanh(1), tanh(Chi4n)), sub(MaxAbsEStateIndex, sub(MaxAbsEStateIndex, HallKierAlpha)), HallKierAlpha, PEOE_VSA7)), Chi4n, EState_VSA1), neg(Chi4n)), NumHeteroatoms, Chi4n)), sub(NumHeteroatoms, tanh(add(NumHeteroatoms, NumHeteroatoms))), PEOE_VSA6), NumHeteroatoms, Chi4n)), NumHeteroatoms), HallKierAlpha), sub(PEOE_VSA6, PEOE_VSA6), EState_VSA1, 0), tanh(MaxAbsEStateIndex), Chi2v), MaxAbsEStateIndex, neg(sub(NumHeteroatoms, MinEStateIndex)), if_lt(if_lt(PEOE_VSA14, sub(tanh(neg(if_lt(cos(MaxAbsEStateIndex), sub(if_lt(FpDensityMorgan1, cos(sub(FpDensityMorgan1, div(if_lt(MinEStateIndex, Kappa3, cos(Chi4n), FpDensityMorgan1), sin(HeavyAtomMolWt)))), Chi4n, EState_VSA1), sub(EState_VSA1, sub(MaxAbsEStateIndex, HallKierAlpha))), NumHeteroatoms, sub(NumHeteroatoms, MinEStateIndex)))), PEOE_VSA6), tanh(VSA_EState9), sub(sub(if_lt(cos(neg(SMR_VSA10)), sin(if_lt(VSA_EState9, sub(MaxAbsEStateIndex, sub(MaxAbsEStateIndex, HallKierAlpha)), cos(neg(SMR_VSA10)), Chi3v)), mul(NumHeteroatoms, if_lt(cos(MaxAbsEStateIndex), if_lt(VSA_EState9, add(FpDensityMorgan1, Chi2v), sub(NumHeteroatoms, tanh(add(mul(EState_VSA1, neg(cos(mul(mul(EState_VSA1, PEOE_VSA7), NumHeteroatoms)))), NumHeteroatoms))), PEOE_VSA6), NumHeteroatoms, Chi4n)), PEOE_VSA14), cos(Chi4n)), cos(Chi2v))), SMR_VSA10, MinEStateIndex, neg(MaxAbsEStateIndex)))
# ```
# 
# *ROC_AUC_SCORE=0.690712*
# 
# ## 3. High Popsize (=1200) and number of Generations (=100) with Smote and MinMax-Scaler
# 
# ```{python}
# 
# ```
# 
# *ROC_AUC_SCORE=*
# 
# 

# # EC2 - Hall of Fame
# 
# ## Basic GP
# ```{python}
# sin(div(div(if_lt(EState_VSA1, mul(VSA_EState9, div(SMR_VSA10, if_lt(fr_COO, VSA_EState9, fr_COO, fr_COO))), sub(NumHeteroatoms, VSA_EState9), sin(tanh(EState_VSA1))), if_lt(fr_COO, add(mul(add(tanh(sin(HallKierAlpha)), fr_COO), fr_COO), sub(mul(VSA_EState9, div(mul(MinEStateIndex, tanh(Chi1v)), div(sin(if_lt(MinEStateIndex, SMR_VSA10, SlogP_VSA3, fr_COO)), Chi1v))), fr_COO)), fr_COO, add(tanh(PEOE_VSA6), PEOE_VSA6))), if_lt(if_lt(cos(div(add(add(EState_VSA1, mul(if_lt(Chi2v, Chi4n, MaxAbsEStateIndex, SMR_VSA5), fr_COO)), sub(div(if_lt(tanh(PEOE_VSA10), sin(if_lt(MinEStateIndex, FpDensityMorgan3, SlogP_VSA3, fr_COO)), sin(HallKierAlpha), sin(tanh(MinEStateIndex))), Chi1v), sub(div(SMR_VSA10, SMR_VSA10), SMR_VSA5))), if_lt(cos(mul(SMR_VSA10, fr_COO)), add(tanh(FpDensityMorgan2), sub(EState_VSA1, PEOE_VSA14)), fr_COO, fr_COO))), add(MinEStateIndex, sub(div(SMR_VSA10, tanh(add(EState_VSA1, mul(SMR_VSA10, fr_COO)))), PEOE_VSA6)), cos(div(add(PEOE_VSA6, if_lt(div(mul(div(mul(mul(tanh(VSA_EState9), sub(-1, Chi2v)), tanh(MinEStateIndex)), neg(FpDensityMorgan2)), PEOE_VSA6), SMR_VSA10), sin(Chi3v), sin(PEOE_VSA7), tanh(Chi4n))), if_lt(div(mul(tanh(PEOE_VSA6), div(sin(if_lt(MinEStateIndex, FpDensityMorgan3, SlogP_VSA3, fr_COO)), fr_COO)), mul(FpDensityMorgan3, fr_COO)), add(tanh(SMR_VSA10), if_lt(if_lt(PEOE_VSA10, HallKierAlpha, SMR_VSA5, fr_COO), add(fr_COO, fr_COO), mul(VSA_EState9, mul(tanh(mul(fr_COO, div(cos(SMR_VSA10), fr_COO))), MinEStateIndex)), fr_COO)), fr_COO, sin(SMR_VSA10)))), VSA_EState9), neg(FpDensityMorgan2), fr_COO, VSA_EState9)))
# ```
# 
# *ROC_AUC_SCORE=0.578397*
# 
# ## ## 2. High Popsize (=1200) and number of Generations (=100)
# 
# 
# ```{python}
# sub(sin(if_lt(add(fr_COO2, Chi3v), FpDensityMorgan1, cos(if_lt(add(if_lt(add(mul(sin(PEOE_VSA8), FpDensityMorgan1), Chi1n), tanh(div(FpDensityMorgan2, div(sin(sin(Kappa3)), cos(EState_VSA1)))), PEOE_VSA6, if_lt(fr_COO2, fr_COO2, FpDensityMorgan3, if_lt(PEOE_VSA10, Chi2n, fr_COO, Chi1n))), Chi1n), tanh(div(FpDensityMorgan2, div(mul(cos(Chi3v), if_lt(PEOE_VSA7, mul(PEOE_VSA10, FpDensityMorgan2), FpDensityMorgan1, if_lt(fr_COO2, tanh(neg(fr_COO)), mul(HeavyAtomMolWt, mul(EState_VSA1, add(fr_COO2, Chi1n))), add(tanh(NumHeteroatoms), if_lt(if_lt(fr_COO2, VSA_EState9, FpDensityMorgan1, if_lt(1, Kappa3, Chi3v, Chi2v)), MaxAbsEStateIndex, fr_COO, NumHeteroatoms))))), cos(EState_VSA1)))), PEOE_VSA6, if_lt(fr_COO2, fr_COO, FpDensityMorgan3, if_lt(PEOE_VSA10, Chi1n, PEOE_VSA14, Chi1n)))), sin(if_lt(add(PEOE_VSA6, add(tanh(div(Chi1n, sin(Chi3v))), fr_COO2)), FpDensityMorgan1, cos(if_lt(tanh(MaxAbsEStateIndex), sub(PEOE_VSA8, mul(neg(mul(add(Chi3v, fr_COO), neg(sin(PEOE_VSA8)))), cos(fr_COO2))), tanh(PEOE_VSA14), fr_COO2)), PEOE_VSA14)))), if_lt(sin(fr_COO), if_lt(div(PEOE_VSA8, cos(PEOE_VSA14)), -1, tanh(add(if_lt(mul(if_lt(NumHeteroatoms, MinEStateIndex, add(if_lt(PEOE_VSA14, cos(EState_VSA1), NumHeteroatoms, MinEStateIndex), Chi3v), fr_COO), fr_COO), sin(fr_COO2), Chi1v, PEOE_VSA14), mul(Kappa3, Chi2n))), fr_COO), if_lt(div(PEOE_VSA7, cos(PEOE_VSA14)), fr_COO2, tanh(ExactMolWt), fr_COO), if_lt(sin(sin(fr_COO)), HallKierAlpha, sin(sin(fr_COO)), fr_COO)))
# ```
# 
# *ROC_AUC_SCORE=0.58231*
# 
# ## 3. High Popsize (=1200) and number of Generations (=100) with Smote and MinMax-Scaler
# 
# ```{python}
# 
# ```
# 
# *ROC_AUC_SCORE=*
