import random
from collections import deque
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras

def huber_loss(y_true, y_pred, clip_delta = 1.0):  # https://en.wikipedia.org/wiki/Huber_loss
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

# Stock trading bot

class Agent:
    def __init__(self, state_size, strategy="t-dqn", model_name=None, test = False,model_path=None):
        self.strategy = strategy
        self.state_size = state_size # no of previous days
        self.action_size = 3 # [hold, buy, sell]
        self.inventory = []
        self.memory = deque(maxlen=1000)
        self.first_iter = True
        
        # model params
        self. model_name = model_name
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        if test == False:
            self.model = self._model()
        else:
            self.custom_objects = {"huber_loss": huber_loss}
            self.model = self.load(model_path)
    def _model(self):
        """
        creating the model
        """
        model = keras.models.Sequential()
        layer1 = keras.layers.Dense(128, activation="relu", input_dim = self.state_size)
        layer2 = keras.layers.Dense(256, activation="relu")
        layer3 = keras.layers.Dense(256, activation="relu")
        layer4 = keras.layers.Dense(128, activation="relu")
        layer5 = keras.layers.Dense(self.action_size)
        model.add(layer1)
        model.add(layer2)
        model.add(layer3)
        model.add(layer4)
        model.add(layer5)
        
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, is_eval=False):
        # choosing random action for exploration
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # ensure we buy in the first iteration
        
        actions_probs = self.model.predict(state)
        return np.argmax(actions_probs[0])
    
    def train_experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        x_train, y_train = [], []
        
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
                
                q_values = self.model.predict(state)
                q_values[0][action] = target
                
                x_train.append(state[0])
                y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()
    
        
        loss = self.model.fit(np.array(x_train), np.array(y_train), epochs=1, verbose = 0).history["loss"][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay
        
        return loss
    
    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))
    
    def load(self, model_path):
        return keras.models.load_model(model_path,custom_objects=self.custom_objects)
        