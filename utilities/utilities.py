from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, log_loss, top_k_accuracy_score


class Utilities:
    @staticmethod
    def evaluator(pipeline, x_train, y_train, x_test, y_test):

        pipeline.fit(x_train, y_train)
        optimized_predictions = pipeline.predict(x_test)
        optimized_preds_probs = pipeline.predict_proba(x_test)
        f1 = f1_score(y_test, optimized_predictions, average='micro')
        loss = log_loss(y_test, optimized_preds_probs)
        acc_3 = top_k_accuracy_score(y_test, optimized_preds_probs, k=3)

        print(f"Optimized Model Acc@3: {acc_3:.4f} -  "
              f"Micro F1-Score performance: {f1:.4f} - "
              f"Loss Value {loss:.4f}")
        return acc_3, f1, loss

    @staticmethod
    def wilcoxon_signed_rank_test(baseline_pipeline, optimized_pipeline,
                                  x_test, y_test,
                                  alpha=0.05):
        # Calculate losses for each instance in the test set for both models
        baseline_probs = baseline_pipeline.predict_proba(x_test)
        baseline_losses = [-log_loss([true_y], [prob],
                                     labels=baseline_pipeline.classes_) for true_y, prob in zip(y_test, baseline_probs)]
        optimized_probs = optimized_pipeline.predict_proba(x_test)
        optimized_losses = [-log_loss([true_y], [prob],
                                      labels=optimized_pipeline.classes_) for true_y, prob in
                            zip(y_test, optimized_probs)]

        # Perform Wilcox signed-rank test to compare the losses
        statistic, p_value = wilcoxon(baseline_losses, optimized_losses)

        print("\nWilcoxon Signed-Rank Test (Log Loss):")
        print(f"Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
        # Interpret the results
        if p_value < alpha:
            print("The optimized pipeline significantly beats the baseline.")
        else:
            print(
                "There is no significant difference between the optimized pipeline and baseline.")

        return statistic, p_value
