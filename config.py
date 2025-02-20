
class Config:
    IMAGE_SIZE = 32
    BATCH_SIZE = 512
    NUM_WORKERS = 0  # set to 2 or more if using Colab/machine with multiple cores
    DATA_ROOT = "data/cifar10"
    CIFAR10_CLASSES = 10

    LR = 0.01
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    NUM_EPOCHS = 50
    NUM_FINETUNE_EPOCHS = 10

    SCAN_STEP = 0.1
    SCAN_START = 0.4
    SCAN_END = 1.0

    SPARSITY_TARGET = (15 / 25)  # Example target sparsity
    SPARSITY_DICT = {} # sparsity based on the layers sensitivity
    DENSE_MODEL_SAVE_PATH = "dense_model.pth"  # Path to save the dense model after training
    FINAL_MODEL_SAVE_PATH = "sparse_model_finetuned.pth"  # Path to save the final sparse model