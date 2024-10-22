# initialize actor and critic networks and their target networks





# initialize replay buffer

# initialize noise process

# set hyperparameters






# get action from actor network

# add noise for exploration

# return action



# store_transition(state, action, reward, next_state, done):











# check if enough experiences in replay buffer, sample a batch for training

# sample a batch from replay buffer randomly

# compute target q-values using target actor and target critic


# compute current q-values using current critic

# compute loss for critic network

# update critic network

# update actor network




# update target networks



# reset noise process at the end of each episode


# save model weights for actor and critic networks to file periodically during training