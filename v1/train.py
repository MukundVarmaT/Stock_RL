import logging
import coloredlogs

from tradingbot_agent import Agent
from tradingbot_methods import train_model, evaluate_model
from tradingbot_utils import get_stock_data, format_currency, format_position, show_train_result, switch_k_backend_device


def main(train_stock, val_stock, window_size, batch_size, ep_count, strategy="dqn", model_name="dqn", debug=False):
    agent = Agent(window_size, strategy=strategy, model_name=model_name)
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)
    
    initial_offset = val_data[1] - val_data[0]
    for episode in range(1, ep_count+1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
        
        
if __name__ == "__main__":

    train_stock = "train_GOOG.csv"
    val_stock = "val_GOOG_2018.csv"
    strategy = "dqn"
    window_size = 10
    batch_size = 32
    ep_count = 100
    model_name = "dqn"
    debug = False

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,ep_count, strategy=strategy, model_name=model_name, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
    