from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class BaselineTrainer:
    @staticmethod
    def train_baseline_models(x_train, y_train, x_test, y_test):
        models = {"RandomForest": RandomForestClassifier(),
                  "DecisionTree": DecisionTreeClassifier(),
                  "LogisticRegression": LogisticRegression(max_iter=1000),
                  "SVM": SVC(probability=True),
                  "KNN": KNeighborsClassifier(),
                  "GradientBoosting": GradientBoostingClassifier()
                  }
        baseline_results = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            preds_probs = model.predict_proba(x_test)
            score = f1_score(y_test, predictions, average='micro')
            acc_3 = top_k_accuracy_score(y_test, preds_probs, k=3)
            loss = log_loss(y_test, preds_probs)
            baseline_results[model_name] = (loss, -score, -acc_3)
            print(f"{model_name} Micro F1-Score: {score:.4f} - Acc@3: {acc_3:.4f} - Log Loss: {loss:.4f}")

        # Select the best model
        best_model_name = min(baseline_results, key=baseline_results.get)
        baseline_results_summary = {'f1': -1 * baseline_results[best_model_name][1],
                                    'acc3': -1 * baseline_results[best_model_name][2],
                                    'log_loss': baseline_results[best_model_name][0]}
        print(
            f"Best Baseline Model: {best_model_name} with \nMicro F1-Score: {-1 * baseline_results[best_model_name][1]:.4f}")
        print(f"Accuracy@3: {-1 * baseline_results[best_model_name][2]:.4f}")
        print(f"Log Loss: {baseline_results[best_model_name][0]:.4f}")
        return best_model_name, models[best_model_name], baseline_results_summary
