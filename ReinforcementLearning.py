
# coding: utf-8

# In[ ]:





# In[7]:


import numpy as np
import gym
import random




# In[9]:


env=gym.make('Taxi-v2')
env.render()


# In[10]:


#Create Q table and initialize it
action_size=env.action_space.n
print('Action Size',action_size)
state_size=env.observation_space.n
print('State Size',state_size)


# In[8]:


q_table=np.zeros((state_size,action_size))
print(q_table)


# In[11]:


#Create hyperparameters
total_episodes =50000    #Total Episodes
total_test_episodes=100 #Total test episodes
max_steps=99            #Max step per episode

learning_rate=0.7        #Learning Rate
gamma=0.618              #Discount Rate


 #Exploration parameters
epsilon=1.0      #Exploration rate
max_epsilon=1.0  #Exploration probability at start
min_epsilon=0.01
decay_rate=0.01


# In[ ]:


#For life or until learning is stopped
for episode in range(total_episodes):
    #Reset the environment
    state=env.reset()
    step=0
    done=False
    
    
    for step in range(max_steps):
        #3.Choose an action a in the current world state
        ##First rendomise a number 
        exp_exp_trade_off=random.uniform(0,1)
        ##if this number>epsilon -->exploitation (taking the biggest Q-value for this state)
        if exp_exp_trade_off > epsilon:
            action=np.argmax(q_table[state,:])
            
        ##Else doing a random choice ------>exploration
        else:
            action=env.action_space.sample()
            
        #Take the action(a) and observe the outcome state(s') and reward(r)
        new_state,reward,done,info=env.step(action)
        #Update Q(s,a):=Q(s,a) +lr[R(s,a)+gamma*maxQ(s',a')-Q(s,a)]
        q_table[state,action]=q_table[state,action]+learning_rate*(reward+gamma*
                                                                  np.max(q_table[new_state,:])-q_table[state,action])
        #Our new state is state
        state=new_state
        
        #If done :finish episode
        if done==True:
            break
            
        episode+=1
        #Reduce epsilon (because we need less and less exploration)
        epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
        


# In[8]:



env.reset()
rewards=[]

for episode in range(total_test_episodes):
    state=env.reset()
    step=0
    done=False
    total_rewards=0
    print('**************************************************************************')
    print('EPISODE',episode)
    
    for step in range(max_steps):
        env.render()
        #Take your action (index) that have the maximum future reward given that state
        action=np.argmax(q_table[state,:])
        new_state,reward,done,info=env.step(action)
        total_rewards=+reward
        
        if done:
            rewards.append(total_rewards)
            print('Score',total_rewards)
            break
        state=new_state
env.close()
print("Score over time"+str(sum(rewards) / total_test_episodes))

