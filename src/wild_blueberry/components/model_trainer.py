import pandas as pd
import os
from wild_blueberry import logger
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import warnings
from pathlib import Path 
warnings.filterwarnings("ignore")
from wild_blueberry.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        extree = ExtraTreesRegressor(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, min_samples_split=self.config.min_samples_split, random_state=42)
        extree.fit(train_x, train_y)

        joblib.dump(extree, os.path.join(self.config.root_dir, self.config.model_name))