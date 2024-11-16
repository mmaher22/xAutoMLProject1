import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utilities.utilities import Utilities


class HyperoptTuning:
    def __init__(self, x_train, y_train):
        self.search = [
            {
                "model_name": "RandomForest",
                "n_estimators": scope.int(hp.quniform("rf-n_estimators", 50, 300, 10)),
                "max_depth": scope.int(hp.quniform("rf-max_depth", 3, 20, 1)),
                "min_samples_split": scope.int(hp.quniform("rf-min_samples_split", 2, 10, 1))
            },
            {
                "model_name": "DecisionTree",
                "max_depth": scope.int(hp.quniform("dt-max_depth", 3, 20, 1)),
                "min_samples_split": scope.int(hp.quniform("dt-min_samples_split", 2, 10, 1))
            },
            {
                "model_name": "LogisticRegression",
                "C": hp.loguniform("lr-C", -4, 2),
                "solver": hp.choice("lr-solver", ["lbfgs", "liblinear"])
            },
            {
                "model_name": "SVM",
                "C": hp.loguniform("svm-C", -4, 2),
                "kernel": hp.choice("svm-kernel", ["linear", "rbf"])
            },
            {
                "model_name": "KNN",
                "n_neighbors": scope.int(hp.quniform("knn-n_neighbors", 3, 20, 1))
            },
            {
                "model_name": "GradientBoostingClassifier",
                "learning_rate": hp.loguniform("gb-learning_rate", -6, 1),
                "n_estimators": scope.int(hp.quniform("gb-n_estimators", 50, 300, 10)),
                "max_depth": scope.int(hp.quniform("gb-max_depth", 3, 20, 1)),
                "min_samples_split": scope.int(hp.quniform("gb-min_samples_split", 2, 10, 1))
            },
        ]
        self.search_space = hp.choice("classifier_type", self.search)
        self.x_train = x_train
        self.y_train = y_train

    def objective(self, params):
        model_name = params["model_name"]
        del params["model_name"]

        if model_name == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(**params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**params, max_iter=1000)
        elif model_name == "SVM":
            model = SVC(probability=True, **params)
        elif model_name == "KNN":
            model = KNeighborsClassifier(**params)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        f1_scores = []
        acc3_scores = []
        log_losses = []

        for train_index, val_index in cv.split(self.x_train, self.y_train):
            x_train_fold, x_val_fold = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            pipeline.fit(x_train_fold, y_train_fold)
            acc3_scores, f1_scores, log_losses = Utilities.evaluator(pipeline,
                                                                     x_train_fold,
                                                                     y_train_fold,
                                                                     x_val_fold,
                                                                     y_val_fold)
            f1_scores.append(f1_scores)
            acc3_scores.append(acc3_scores)
            log_losses.append(log_losses)

        mean_f1 = np.mean(f1_scores)
        mean_acc3 = np.mean(acc3_scores)
        mean_loss = np.mean(log_losses)

        return {"loss": mean_loss,
                "status": STATUS_OK,
                "log_loss": mean_loss,
                "f1-metric": mean_f1,
                "acc3-metric": mean_acc3}

    def optimize_hyperopt(self):
        trials = Trials()
        best_hyperparams = fmin(fn=self.objective,
                                space=self.search_space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

        print("Best Hyperparameters:", best_hyperparams)
        return best_hyperparams

    def get_best_model_from_hyperparams(self, best_hyperparams):
        model_name = self.search[best_hyperparams["classifier_type"]]["model_name"]

        params = {k.split('-')[-1]: v for k, v in best_hyperparams.items()
                  if k != "classifier_type" and k != "model_name"}

        if 'solver' in params.keys():
            params['solver'] = 'lbfgs' if params['solver'] == 0 else 'liblinear'
        elif 'kernel' in params.keys():
            params['kernel'] = 'linear' if params['kernel'] == 0 else 'rbf'

        for param_name in ["max_depth", "n_estimators", "min_samples_split", "n_neighbors"]:
            if param_name in params:
                params[param_name] = int(params[param_name])

        if model_name == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(**params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**params, max_iter=1000)
        elif model_name == "SVM":
            model = SVC(**params, probability=True)
        elif model_name == "KNN":
            model = KNeighborsClassifier(**params)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return Pipeline([("scaler", StandardScaler()), ("classifier", model)])