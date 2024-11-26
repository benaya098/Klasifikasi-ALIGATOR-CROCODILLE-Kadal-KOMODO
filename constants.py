TRAIN_DIR = r'C:\Users\CERTAN\'
TEST_DIR = r'C:\Users\CERTAN\'

# Labels for dataset
Buaya_LBL = 0
Alligator_LBL = 1
Komodo_LBL = 2
Cicak_LBL = 3

Buaya = 'Buaya'
Alligator = 'Alligator'
Komodo = 'Komodo'
Cicak = 'Cicak'

LABEL_MAP = {
    Buaya: Buaya_LBL,
    Alligator: Alligator_LBL,
    Komodo: Komodo_LBL,
    Cicak: Cicak_LBL
}

# Other parameters
DATA_SIZE = 18_000  # Total dataset used
IMG_SIZE = 110  # Size of images to be resized
SPLIT_RATIO = 0.8  # Ratio for splitting training and testing data
