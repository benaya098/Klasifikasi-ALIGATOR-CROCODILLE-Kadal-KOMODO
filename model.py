from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def get_model():
    model = Sequential([
        Input(shape=(110, 110, 3)),  
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.4),

        Dense(4, activation='softmax') 
    ])

    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model