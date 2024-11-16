import os
import random
import numpy as np
import pandas as pd
from controllers.hyperopt_tuning import HyperoptTuning
from controllers.baseline_trainer import BaselineTrainer
from utilities.utilities import Utilities

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)


if __name__ == '__main__':
    # Step 0: Load and split dataset
    train_data = pd.read_csv('knowledge_base/Knowledge_base_train.csv')
    test_data = pd.read_csv('knowledge_base/Knowledge_base_test.csv')

    # Separate features and target
    x_train = train_data.drop(columns=["class"])
    y_train = train_data["class"]
    x_test = test_data.drop(columns=["class"])
    y_test = test_data["class"]

    # Step 1: Build and train baseline models with default hyperparameters
    (best_baseline_name,
     best_baseline_model,
     baseline_results) = BaselineTrainer.train_baseline_models(x_train, y_train, x_test, y_test)

    # Step 2: Optimize the hyperparameters
    hyperopt_tuner = HyperoptTuning(x_train, y_train)
    best_hyperparams = hyperopt_tuner.optimize_hyperopt()
    optimized_model_pipeline = hyperopt_tuner.get_best_model_from_hyperparams(best_hyperparams)

    # Step 3: Wilcoxon Signed Rank Test
    statistic, p_value = Utilities().wilcoxon_signed_rank_test(best_baseline_model,
                                                               optimized_model_pipeline,
                                                               x_test, y_test)
