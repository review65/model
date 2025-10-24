# demand_model.py (FIXED VERSION)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

def build_lstm_model(input_shape):
    """
    สร้างสถาปัตยกรรมโมเดล LSTM ที่ปรับปรุงแล้ว
    พร้อม Regularization เพื่อป้องกัน Overfitting
    """
    model = Sequential()
    
    # Layer 1: Bidirectional LSTM (เรียนรู้ทั้ง forward และ backward)
    model.add(Bidirectional(
        LSTM(units=64, 
             activation='tanh',
             recurrent_activation='sigmoid',
             input_shape=input_shape,
             return_sequences=True,
             kernel_regularizer=l2(0.001),
             recurrent_regularizer=l2(0.001))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Layer 2: LSTM
    model.add(LSTM(units=32,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   return_sequences=False,
                   kernel_regularizer=l2(0.001),
                   recurrent_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Dense Layers
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    
    # --- !! แก้ไข !! ---
    # Output Layer
    # เพิ่ม activation='relu' เพื่อบังคับให้โมเดลไม่ทายค่าติดลบ
    model.add(Dense(units=1, activation='relu')) 
    
    # Compile with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        # --- !! แก้ไข !! ---
        # เปลี่ยนเป็น 'mean_absolute_error' (MAE) 
        # เพื่อให้ทนทานต่อ Outliers (เช่น ยอดขาย 17) ได้ดีกว่า MSE
        loss='mean_absolute_error', 
        metrics=['mae']
    )
    
    print("\n" + "="*60)
    print("LSTM MODEL ARCHITECTURE (Using MAE Loss + ReLU Output)") # <--- แก้ไข Title
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    return model


def build_alternative_model(input_shape):
    """
    โมเดลทางเลือก: GRU-based (เร็วกว่า LSTM)
    """
    from tensorflow.keras.layers import GRU
    
    model = Sequential()
    
    # Layer 1: Bidirectional GRU
    model.add(Bidirectional(
        GRU(units=64,
            activation='tanh',
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Layer 2: GRU
    model.add(GRU(units=32,
                  activation='tanh',
                  return_sequences=False,
                  kernel_regularizer=l2(0.001),
                  recurrent_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Dense Layers
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    
    # --- !! แก้ไข (เผื่อใช้) !! ---
    # Output
    model.add(Dense(units=1, activation='relu')) # <--- แก้ไข
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error', # <--- แก้ไข
        metrics=['mae']
    )
    
    print("\n" + "="*60)
    print("GRU MODEL ARCHITECTURE (Using MAE Loss + ReLU Output)") # <--- แก้ไข
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    return model