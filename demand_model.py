# demand_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    สร้างสถาปัตยกรรมโมเดล LSTM สำหรับพยากรณ์อุปสงค์
    """
    model = Sequential()
    
    # --- เติมส่วนสถาปัตยกรรมโมเดล ---
    # Layer ที่ 1
    # 'return_sequences=True' เพราะเราจะส่งต่อไปยัง LSTM Layer ที่สอง
    model.add(LSTM(units=64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Layer ที่ 2
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer
    # เราพยากรณ์ค่าเดียวคือ 'Total_Qty'
    model.add(Dense(units=1))
    # --- สิ้นสุดส่วนที่เติม ---
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\nLSTM model built successfully.")
    model.summary()
    
    return model