from pathlib import Path
# import yaml

from transformers_for_timeseries.ml_tools.random_seeds_setting import seed_everything
from transformers_for_timeseries.ml_tools.pytorch_device_selection import select_torch_device

class Config:
    """Global variables configuration"""

    NAME = "synthetic_pendulum_1"

    ##########################################    
    ### Paths
    DIR_DATA = Path(__file__).absolute().parent.parent.parent.parent / "data"
    DIR_RAW_DATA = DIR_DATA / f"raw" / f"{NAME}"
    DIR_PREPROCESSED_DATA = DIR_DATA / f"preprocessed" / f"{NAME}"
    DIR_PROCESSED_DATA = DIR_DATA / f"processed" / f"{NAME}"
    DIR_OTHERS_DATA = DIR_DATA / f"others" / f"{NAME}"
    DIR_RAW_DATA.mkdir(parents=True, exist_ok=True)
    DIR_PREPROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    DIR_PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    DIR_OTHERS_DATA.mkdir(parents=True, exist_ok=True)

    DIR_RESULTS = Path(__file__).absolute().parent.parent.parent.parent / f"results" / f"{NAME}"
    DIR_MODEL_PARAMS = DIR_RESULTS / f"model_params"
    DIR_PARAMS_CHECKPOINTS = DIR_MODEL_PARAMS / f"checkpoints"
    DIR_FIGURES = DIR_RESULTS / f"figures"
    DIR_MODEL_PARAMS.mkdir(parents=True, exist_ok=True)
    DIR_PARAMS_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)


    ### PyTorch device & set seed for reproducibility
    DEVICE = select_torch_device(temporal_dim="sequential", gpu_id=1)
    SEED = 42
    seed_everything(SEED)

    ### Data parameters
    # Synthetic forced damped pendulum
    N_SAMPLES = 5000
    MAXTIME = 100.0
    TIMESTEPS = 2000

    BETA_RANGE = (0.1, 1.0)
    KAPA_RANGE = (0.5, 20.0)
    THETA0 = 1.0
    OMEGA0 = 0.0
    A_VALUE = None
    OMEGA_VALUE = 1.0

    ### DataLoader parameters
    SPLIT_RATIO = [0.7, 0.15, 0.15]   # train, valid, test
    TRAIN_SIZE = int(N_SAMPLES * SPLIT_RATIO[0])
    VALID_SIZE = int(N_SAMPLES * SPLIT_RATIO[1])
    TEST_SIZE = N_SAMPLES - TRAIN_SIZE - VALID_SIZE

    ### SCINET architecture
    M_INPUT_SIZE = 1280     # output of Transformer encoder
    M_ENC_HIDDEN_SIZES =  [500, 500]
    M_LATENT_SIZE = 3
    M_QUESTION_SIZE = TIMESTEPS
    M_DEC_HIDDEN_SIZES =  [500, 500, 500]
    M_OUTPUT_SIZE = TIMESTEPS

    ### Hyperparameters
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_EVAL = 100
    FIRST_LEARNING_RATE = 3e-3
    WEIGHT_DECAY = 1e-6     # if needed
    KLD_BETA = 0.0002

    ### Train parameters
    NUM_EPOCHS = 150
    ES_PATIENCE = 12
    ES_MIN_DELTA = 5e-5
    GC_MAX_NORM = 1.0
    LRS_FACTOR = 0.66
    LRS_PATIENCE = 6
    LRS_MIN_LR = 1e-7
    LRS_MIN_DELTA = 1e-5

    ### Others
    BEST_MODEL_NAME = "pendulum_scinet_final"

   
    # Method to update parameters
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# Global instance
config = Config()
