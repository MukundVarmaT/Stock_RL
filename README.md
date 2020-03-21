# STOCK TRADING BOT

My implementation of a Stock trading bot using reinforcement learning.

The Data is obtained from [Yahoo Finance](https://in.finance.yahoo.com/) where one can easily export historical transactions between two dates into csv files.

I have trained the model on **Alphabet Inc.** stocks from 2010 to 2017 and hence tested on 2019 stock prices.

The model was able to get a profit of almost **$2500** for the year 2019. Now this is a good model considering the fact that the model has the capability of only buying/selling only one stock and the maximum price in that year is **less than $1200.**

I have also created a **dash application** to see the decisions made by the bot at a particular time and the total profit earned till then. A speeded up video of the same.

![dash](images/stock-DQN.gif)


## Model-specific details:

- Action size = 3 [ buy, sell, hold]
- Reward policy = profit earned when selling a stock when compared to the price at which it bought it and otherwise 0.
- gamma = 0.95
- epsilon = 1.0
- epsilon_min = 0.01
- epsilon_decay = 0.995
- learning_rate = 0.001

### 21st March 2020 Update:

**A new reward policy**

reward =
- profit earned when selling a stock when compared to the price at which it bought it
- if the agent has bought something then the difference between the bought price and the current price when it decides to HOLD and if everything in hand is sold then the difference between the price right now and the price the next day. (which is the opportunity cost of buying a stock right now and selling it immediately in the next state)

Currently the model achieves a higher profit more than **$3450** in the year 2019.

## To do

- [x] Vanilla DQN
- [x] Change the reward policy.
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Ability of the bot to purchase multiple quantity of stocks.
- [ ] Ability of the bot to choose from different stock options.

