# demand_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    สร้างสถาปัตยกรรมโมเดล LSTM สำหรับพยากรณ์อุปสงค์
    """
    model = Sequential()
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\nLSTM model built successfully.")
    model.summary()
    
    return model