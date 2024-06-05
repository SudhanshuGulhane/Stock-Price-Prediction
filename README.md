### Note
Do not use this to predict real time stock prices or use it for trading. The model does not consider the impact of various factors like sentiments, articles, news, politics on the stock prices.
Also, personally I feel that predicting the closing price of a stock isn't a good idea, cause in the end we are interested in the future returns. Also taking stock price as input may not be a good idea, if the 
data is not stationary. Additionally, there can be one test where we expect our model to predict the future stock price in a certain price range, for which we did not train our model. I plan to work on these 
for the future work and make my model more robust.

# The primary objective of this project
To develop a LSTM model that can accurately predict the closing prices of stocks, and importantly to learn how they function. The dataset comprises historical stock prices of multiple companies categorized into three volatility segments: low, stable, and high. The specific goals of this project are:

**Data Segmentation**: Segment the stock data into three categories based on volatility: low, stable, and high.

**Feature Engineering**: Enhance the dataset with technical indicators such as Relative Strength Index (RSI) and Exponential Moving Averages (EMAs).

**Model Training**: Train separate LSTM models for each volatility category to predict future stock prices.

**Evaluation**: Evaluate the performance of the models using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

**Visualization**: Aggregate the results yearly and visualize the predictions against true values to interpret the model's performance.

### Data Preparation

**Segmentation**: The stock data is divided into three categories based on annualized volatility. The volatility is calculated using the daily closing prices.

**Feature Engineering**: Added technical indicators like RSI and EMAs to the dataset to provide additional features for the model.

**Scaling**: The features are scaled using MinMaxScaler to normalize the input data.

### Model Architecture

LSTM Network: The model uses an LSTM network with the following parameters:
Input Dimension: 8
Hidden Dimension: 64
Output Dimension: 1
Number of Layers: 2
Dropout: 0.5

### Training and Evaluation

**Data Loaders**: Data is split into training and testing sets with an 80-20 split. DataLoader is used to handle batching.

**Loss Function**: Mean Squared Error (MSE) is used as the loss function.

**Optimizer**: AdamW optimizer with learning rate 0.001 and weight decay 1e-4.

**Scheduler**: ReduceLROnPlateau scheduler to adjust the learning rate based on validation loss.

**Early Stopping**: Implemented to stop training if the validation loss does not improve for a specified number of epochs.

### Implementation

**Code Structure**: The implementation includes functions for data preparation, model training, evaluation, and plotting.

**Libraries**: The project leverages libraries like Pandas, NumPy, PyTorch, and Matplotlib for data handling, model training, and visualization.

### Most Interesting Results

The models for each volatility category (low, stable, high) were trained separately.
Evaluation metrics were collected, and the predictions were visualized against true values.

# Plots

### High Volatile Stocks

![Actuals and Predictions](https://github.com/SudhanshuGulhane/Stock-Price-Prediction/assets/50482460/99d206ad-aef8-4ce9-b505-9a497ed7129f)

### Low Volatile Stocks

![Actuals and Predictions](https://github.com/SudhanshuGulhane/Stock-Price-Prediction/assets/50482460/55e00516-e06a-42db-9658-ac6cd9c421d8)

### Stable Volatile Stocks

![Actuals and Predictions](https://github.com/SudhanshuGulhane/Stock-Price-Prediction/assets/50482460/39fb76c7-53e4-42d7-a471-f495103bd12d)

### Complete Stocks Data

![predictions and actuals](https://github.com/SudhanshuGulhane/Stock-Price-Prediction/assets/50482460/9f8951d6-a9ed-413c-b668-374b3c49f38a)

