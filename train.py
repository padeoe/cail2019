import logging
import os

from model import HyperParameters, BertModelTrainer

logger = logging.getLogger("train model")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
fh = logging.FileHandler(os.path.join(MODEL_DIR, "train.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == "__main__":
    BERT_PRETRAINED_MODEL = "/bert/pytorch_chinese_L-12_H-768_A-12"

    TRAINING_DATASET = "data/train/input.txt"  # for quick dev
    # TRAINING_DATASET = "data/raw/CAIL2019-SCM-big/SCM_5k.json"

    test_input_path = "data/test/input.txt"
    test_ground_truth_path = "data/test/ground_truth.txt"

    config = {
        "max_length": 512,
        "epochs": 2,
        "batch_size": 12,
        "learning_rate": 2e-5,
        "fp16": False,
        "fp16_opt_level": "O1",
        "max_grad_norm": 1.0,
        "warmup_steps": 0.1,
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = config
    algorithm = "BertForSimMatchModel"

    trainer = BertModelTrainer(
        TRAINING_DATASET,
        BERT_PRETRAINED_MODEL,
        hyper_parameter,
        algorithm,
        test_input_path,
        test_ground_truth_path,
    )
    trainer.train(MODEL_DIR, 1)
