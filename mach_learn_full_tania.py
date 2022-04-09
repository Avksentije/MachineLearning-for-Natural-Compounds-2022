#!/usr/bin/env python
# coding: utf-8

# FIRST PART
# 
# The first half of this notebook is the same as day3 file.
# The second half: machine learning and visuals.

# In[8]:


import pandas as pd
import numpy as np
import os


# In[9]:


ls


# In[10]:


x=pd.read_csv("erk2_labelled_binary.smi", sep="\t", header=None)


# In[131]:


x.head()


# In[138]:


x.columns="smiles","label"


# In[139]:


x["label"].value_counts()


# In[12]:


import rdkit.Chem as Chem
import rdkit.Chem.AllChem



from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize
rdBase.rdkitVersion


# In[13]:


Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in x.loc[0:4,0]])


# In[14]:


Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in x.loc[14520:,0]])


# In[17]:


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize
rdBase.rdkitVersion


# In[18]:


s=Chem.SmilesMolSupplier("erk2_labelled_binary.smi", delimiter="\t", titleLine=False)


# In[19]:


morgan_fp=[AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in s if mol is not None]


# In[20]:


morgan_fp


# In[ ]:


morgan_fp=np.asarray(morgan_fp, dtype=np.int)


# In[30]:


ids=[mol.GetProp("_Name") for mol in s if mol is not None]


# In[31]:


labels=np.asarray(ids, dtype=np.int)


# In[32]:


labels


# In[33]:


labels_row=np.asarray(ids, dtype=np.int).reshape(-1,1)


# In[34]:


labels_row


# In[97]:


labels_row.shape


# In[98]:


morgan_fp.shape


# In[99]:


combined_fp=np.concatenate([morgan_fp, labels_row], axis=1)


# In[100]:


combined_fp[0:, 0:2048]


# In[101]:


combined_fp.shape


# In[102]:


features=combined_fp[0:, 0:2048]


# In[103]:


features


# In[44]:


np.save("erk2_morgan_fp_mach", combined_fp)


# In[45]:


csv=pd.DataFrame(combined_fp)


# In[46]:


csv


# In[47]:


csv.to_csv("morgan_fp_mach.csv")


# In[104]:


labels_row


# SECOND PART:
# 
# 
# Now we begin the machine learning coding
# 
# We are going to need to import some libraries to train our algorithm:
# 
# 

# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix


# Train and Test
# 
# Our parameters of training and test are going to be:
# 
# Training dataset = 80% of total (14525)
# Test dataset = 20% of total (14525)
# 
# 
# They depend on the database size, but the general rule is 75% and 25%

# In[55]:


features.shape


# In[105]:


labels_row.shape


# In[106]:


X_train, X_test, y_train, y_test=train_test_split(features, labels_row, test_size=0.25,random_state=42, shuffle=True, stratify=labels)


# if we use random state, we may have the problem the academic community may not replicate our work
# we fix it using a random_state = 42


# So our total rows are 14525. 
# We are working with the 75% for the train (10893 rows) and the 25% for the test (3632 rows), randomly used (suffle=True).

# In[107]:


X_train.shape


# In[108]:


X_test.shape


# In[109]:


rf=RandomForestClassifier()

# The random forest is the tree of decision-making to stimate the mach learning
# By default there are 100 trees, that's why we leave the parenthesis empty
# In many cases 10 trees have been enough. 
# It is a parameter to take into account because it defines the speed of the process (training and testing)


# In[110]:


rf.fit(X_train, y_train)


# In[111]:


y_train


# In[112]:


y_test


# In[113]:


predicted=rf.predict(X_test)


# In[114]:


predicted


# In[142]:


predicted.shape


# In[115]:


roc_auc_score(y_test, predicted)


# The 0.7476 doesn't tell us how accurate is our training, so we use a Confusion Matrix to figure that out.

# In[118]:


from pycm import *
cm = ConfusionMatrix(y_test.reshape(-1), predicted)


# In[119]:


print(cm)


# In[ ]:


.
.
.
# Now we are going to find similarities in the fingerprints and virtual screening


# In[120]:


import rdkit.DataStructs


# In[121]:


rdkit.DataStructs.TanimotoSimilarity

# We compare the similarity on the structure of a molecule compare with one(s) another(s) using the Tanimoto Similarity


# In[ ]:


tanimoto=[]
morgan_fp_ts=[AllChem.GetMorganFingerprint(mol, 2) for mol in s if mol is not None]
for i, fp in enumerate(morgan_fp_ts[0:5]):
    for j, fp1 in enumerate(morgan_fp_ts[0:5]):
        tanimoto.append(round(rdkit.DataStructs.TanimotoSimilarity(fp,fp1),3))


# In[311]:


pd.DataFrame(tanimoto)


# In[312]:


x=pd.read_csv("erk2_labelled_binary.smi", sep="\t", header=None)


# In[313]:


x.columns="smiles","label"


# In[314]:


x["label"].value_counts()


# VIRTUAL SCREENING
# 
# Bioactivity
# QSAR : Cuantitative structure-activity relationship models
# 
# ERK2 kinase>>> Try to identify some newer Natural Compounds for the ERK2 Kinase based on our random forest model built on Morgan fingerprint

# Shallow/Classical machine learning
# 
# Deep neural networks (epoch)
# 
# Training 80% of the data
# Testing 20% of the data
# Validation 10% of the data
# 

# In[315]:


ls


# In[408]:


copy acdiscnp_p0.smi+ibsnp_p0.smi+afronp_p0.smi full_com.smi


# In[409]:


ls


# In[410]:


np_full=Chem.SmilesMolSupplier("full_com.smi", delimiter=" ", titleLine=False)

# Here I am going to work with the full data base of the three db merged, which I named full_com.smi


# In[421]:


np_full


# In[323]:


morgan_np_full=[AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in np_com if mol is not None]


# In[324]:


morgan_np_full=np.array(morgan_np_full, dtype=np.int)


# In[327]:


morgan_np_full


# In[328]:


morgan_np_full.shape


# In[422]:


ids_full=[mol.GetProp("_Name") for mol in np_full if mol is not None]


# In[426]:


new_prediction_full = rf.predict(morgan_np_full)

#this is the screening using our model rf


# In[427]:


new_prediction_full


# In[428]:


pd.DataFrame(new_prediction_full)


# In[429]:


np_predicted_full_zinc = pd.DataFrame(new_prediction_full)


# In[430]:


np_active_full = np_predicted_full_zinc[np_predicted_full_zinc[0] == 1]


# In[431]:


np_active_full


# In[432]:


zinc_ids_full=pd.DataFrame(ids_full)


# In[433]:


zinc_ids_full


# In[434]:


outcome=pd.concat([zinc_ids_full, np_predicted_full_zinc], axis=1) 


# In[435]:


outcome


# In[436]:


outcome.columns="zinc_id", "prediction"


# In[437]:


outcome


# In[438]:


id_active_full = outcome[outcome["prediction"] == 1]


# In[439]:


id_active_full


# At this point, we have a general result of active 0 and inactive 1
# 
# But we want to be more sensitive, so we run again our model, but we are going to obtain an array of probabilities

# In[441]:


new_prediction_proba_full = rf.predict_proba(morgan_np_full)


# In[442]:


new_prediction_proba_full


# In[443]:


new_prediction_proba_full.shape


# In[444]:


new_score_full=pd.DataFrame(new_prediction_proba_full)


# In[445]:


new_score_full


# In[447]:


more_proba_full=pd.concat([zinc_ids_full, new_score_full], axis=1) 


# In[448]:


more_proba_full


# In[458]:


more_proba_full.columns="zinc_id", "inactive", "active"


# And now, what we do, is to made a new frame only with a probability of activity higher than 0.9

# In[471]:


id_active_proba_full = more_proba_full[more_proba_full["active"]>0.9]


# In[472]:


id_active_proba_full.shape


# In[473]:


id_active_proba_full


# In[474]:


#we have the 22 molecules with more probable positive bioactivity


# In[475]:


csv=pd.DataFrame(id_active_proba_full)


# In[476]:


csv


# In[488]:


csv.shape


# In[477]:


csv.to_csv("pure_happiness_full.csv")


# And we look for the structure, so we open the original data base "ibsnp_p0.smi"
# 
# So we match our id_active_proba with the zinc_id of the original data base and then we visualize the molecule :)

# In[493]:


main_data=pd.read_csv("full_com.smi", header=None, sep=" ")


# In[494]:


main_data.columns = "smiles", "zinc_id"


# In[495]:


main_data


# In[496]:


common=pd.merge(main_data,id_active_proba_full, on="zinc_id", indicator=True)


# In[497]:


common


# In[491]:


common.shape


# In[508]:


common_unique=common.drop_duplicates("zinc_id", keep="first")


# In[509]:


common_unique


# In[510]:


common_unique.shape


# In[511]:


common_unique[["smiles", "zinc_id"]].to_csv("pure_happiness_full.smi", sep="\t", header=None)


# In[512]:


common_unique = common_unique[["smiles", "zinc_id"]]


# In[513]:


Draw.MolsToGridImage([Chem.MolFromSmiles(common) for common in common.iloc[0:19,0]])


# 
# Machine Learning Course by Dr. Pankaj and Dr Aaron
# 
# Documented by Tania R Calzada

# In[514]:


from padelpy import padeldescriptor


# In[ ]:


common_unique


# In[ ]:





# In[ ]:





# In[ ]:




