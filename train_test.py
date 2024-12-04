from data_prep import ImageDataLoader
from model import get_model

EPOCHS = 5  # Jumlah epoch untuk pelatihan

def train_and_evaluate():
    # Path dataset
    data_loader = ImageDataLoader(train_dir=r"C:\Users\CERTAN\train", test_dir=r"C:\Users\CERTAN\test1")

    # Load data
    X_train, X_val, y_train, y_val = data_loader.load_data()
    X_test, filenames = data_loader.load_test_data()

    # Load model
    model = get_model()

    # Train model
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))

    # Predict test data
    predictions = model.predict(X_test)
    return history, predictions, filenames
