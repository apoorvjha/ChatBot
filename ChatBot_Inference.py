#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ChatBot_Utility as bot


# In[2]:


bot.warnings.filterwarnings('ignore')


# In[3]:


with open("input.json","r") as file:
    data1=bot.json.load(file)
with open("intents.json","r") as file:
    data2=bot.json.load(file)


# In[4]:


X=bot.inputPreprocessing(data1,data2)


# In[5]:


X=bot.np.array(X)


# In[6]:


with open("tags.json",'r') as file:
   data=bot.json.load(file)


# In[7]:


model=bot.Model(100,16,30,10,fname='model.h5')


# In[8]:


model.load_model('model.h5')


# In[9]:


preds=model.predict(X)


# In[10]:


res=[]
for i in preds:
    for j in data2['intents']:
        if j['tag']==data[str(bot.np.argmax(i))]:
            res.append([j['responses'][bot.random.randint(0,len(j['responses'])-1)],j['action']])


# In[11]:


response_dict={"responses" : [],"action" : []}
for i in res:
    response_dict["responses"].append(i[0])
    response_dict["action"].append(i[1])
with open("response.json",'w') as file:
    bot.json.dump(response_dict,file)


# In[ ]:




