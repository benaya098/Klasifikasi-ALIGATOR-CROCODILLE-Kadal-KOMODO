from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

def get_model():
    model = Sequential([
        Input(shape=(110, 110, 3)),
        Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.2),
        Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
