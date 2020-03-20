"""
Script for training STOCK TRADING BOT

USAGE:

python3 train.py <trainstockfile> <valstockfile> [--strategy=<strategy>] [--window-size=<window-size>] [--episode-count=<episode-count> [--model-name=<model-name>] [--pretrained] [--debug]]


OPTIONS:

--strategy      :   Q Learning strategy used to train the network.
                    Options: dqn        - Vanilla DQN
                            t-dqn       - DQN with fixed target distribution
                            double-dqn  - DQN with seperate network for value estimation
--window-size:  :   size of the n day window stock data used as feature vector
--batch-size    :   Number of samples to train on in one mini-batch
--episode-count :   Number of trading episodes to use for training 
--model-name    :   Name of pretrained model to use.
--pretrained    :   To choose whether or not we must use a pretrained model.
--debug         :   Specifies whether to use verbose logs during eval operation.

"""

import logging
import coloredlogs
from docopt import docopt

from tradingbot_agent import Agent
from tradingbot_methods import train_model, evaluate_model
from tradingbot_utils import get_stock_data, format_currency, format_position, show_train_result, switch_k_backend_device


def main(train_stock, val_stock, window_size, batch_size, ep_count, strategy="t-dqn", model_name="t-dqn", pretrained=False, debug=False):
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)
    
    initial_offset = val_data[1] - val_data[0]
    for episode in range(1, ep_count+1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
        
        
if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,ep_count, strategy=strategy, model_name=model_name,pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
    