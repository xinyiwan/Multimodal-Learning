import json
import os
import pandas as pd
import argparse
import numpy as np
import scipy.stats as st
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')


LABEL_DIS = '/projects/prjs1779/Osteosarcoma/exp_data/label_distributions_summary.csv'


def combine_probabilities_by_subject(csv_path, threshold=0.5):
    """
    Combine image-level predictions to patient-level by averaging probabilities.
    sample_id format: OS_000002_01 -> patient_id: OS_000002
    Expects columns: sample_id, y_true, y_prob_ensemble, n_models.
    """
    try:
        df = pd.read_csv(csv_path)

        # sample_id is image-level: e.g. OS_000002_01 -> patient OS_000002
        df['patient_id'] = df['sample_id'].apply(lambda x: '_'.join(x.split('_')[:2]))

        combined_df = df.groupby('patient_id').agg(
            y_true=('y_true', 'first'),
            y_prob_ensemble=('y_prob_ensemble', 'mean'),
        ).reset_index()

        combined_df['prediction'] = (combined_df['y_prob_ensemble'] >= threshold).astype(int)
        return combined_df
    except Exception as e:
        print(f"Error combining probabilities by subject: {e}")
        return None
        
def calculate_metrics(predictions_df):
    """Calculate metrics from subject-level predictions (y_true / y_pred_ensemble columns)."""
    probs = predictions_df['y_prob_ensemble'].values
    labels = predictions_df['y_true'].values

    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in data")
        return {
            'auroc': np.nan,
            'accuracy': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'n_samples': len(labels),
            'n_positive': int(sum(labels)),
            'n_negative': len(labels) - int(sum(labels)),
            'tp': np.nan, 'tn': np.nan, 'fp': np.nan, 'fn': np.nan
        }

    preds = (probs > 0.5).astype(int)

    try:
        auroc = roc_auc_score(labels, probs)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auroc = np.nan

    accuracy = accuracy_score(labels, preds)

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        tp = tn = fp = fn = 0
        sensitivity = specificity = np.nan

    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'n_samples': len(labels),
        'n_positive': int(tp + fn),
        'n_negative': int(tn + fp),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def extract_from_one_fold(res_path, modality, fold_num):
    fold_path = res_path / modality / fold_num
    best_trial_info = fold_path / "current_best_test_preds" / "best_trial_info.json"
    if not best_trial_info.exists():
        print(f"Warning: {best_trial_info} does not exist. Skipping.")
        return None
    with open(best_trial_info, "r") as f:
        best_trial_data = json.load(f)
        best_trial_num = best_trial_data.get("trial_number", {})
        best_trial_auc_dev = best_trial_data.get("AUC_dev", {})
        best_trial_auc_test = best_trial_data.get("test_auc_ensemble", {})

    best_trial_path = fold_path / f"trial0{best_trial_num}_fold{fold_num}"
    test_preds_ensemble = best_trial_path / "test_preds_ensemble.csv"
    if not test_preds_ensemble.exists():
        print(f"Warning: {test_preds_ensemble} does not exist. Skipping.")
        return None

    combined_preds_df = combine_probabilities_by_subject(test_preds_ensemble)
    if combined_preds_df is None or combined_preds_df.empty:
        print(f"Warning: Could not combine predictions for fold {fold_num}. Skipping.")
        return None

    metrics = calculate_metrics(combined_preds_df)

    return {
        "fold": int(fold_num),
        "auroc": metrics['auroc'],
        "accuracy": metrics['accuracy'],
        "sensitivity": metrics['sensitivity'],
        "specificity": metrics['specificity'],
        "n_samples": metrics['n_samples'],
        "n_positive": metrics['n_positive'],
        "n_negative": metrics['n_negative'],
        "tp": metrics['tp'],
        "tn": metrics['tn'],
        "fp": metrics['fp'],
        "fn": metrics['fn'],
        "best_trial_num": best_trial_num,
        "best_trial_auc_dev": best_trial_auc_dev,
        "best_trial_auc_ensemble": best_trial_auc_test,
    }


def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval for cross-validation.
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 0.95
    """

    # Remove NaN values if they are there
    if np.isnan(metric).any():
        print('[WORC Warning] Array contains nan: removing.')
        metric = np.asarray(metric)
        metric = metric[np.logical_not(np.isnan(metric))]

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI


def generate_roc_with_ci(res_path, modality, alpha=0.95, n_samples=20):
    """
    Generate ROC curve data with confidence intervals across folds.
    Uses WORC-style threshold sampling: collects all thresholds from ROC curves
    and samples them intelligently.

    Returns a DataFrame with FPR and TPR ranges for each threshold.
    """
    all_fpr = []
    all_tpr = []
    all_thresholds = []

    for fold in range(5):
        fold_num = str(fold)
        fold_path = res_path / modality / fold_num
        best_trial_info = fold_path / "current_best_test_preds" / "best_trial_info.json"
        if not best_trial_info.exists():
            continue

        with open(best_trial_info, "r") as f:
            best_trial_data = json.load(f)
            best_trial_num = best_trial_data.get("trial_number", {})

        test_preds_ensemble = fold_path / f"trial0{best_trial_num}_fold{fold_num}" / "test_preds_ensemble.csv"
        if not test_preds_ensemble.exists():
            continue

        df = combine_probabilities_by_subject(test_preds_ensemble)
        if df is None or df.empty:
            continue

        if len(np.unique(df['y_true'])) == 2:
            fpr, tpr, thresholds = roc_curve(df['y_true'].values, df['y_prob_ensemble'].values)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_thresholds.append(thresholds)

    if not all_fpr:
        print("No valid predictions found for ROC curve generation")
        return None

    print(f"Found {len(all_fpr)} valid folds for ROC curve generation")

    # Get sample sizes for CI calculation
    distribution_df = pd.read_csv(LABEL_DIS)
    median_n_train = int(np.mean(distribution_df[distribution_df['modality'] == modality]['train_total'].values))
    median_n_test = int(np.mean(distribution_df[distribution_df['modality'] == modality]['test_total'].values))

    # WORC-style threshold sampling: combine all thresholds and sample indices
    T = []
    for t in all_thresholds:
        T.extend(t)
    T = sorted(T, reverse=True)  # Sort in descending order (high to low threshold)

    # Sample indices uniformly across the combined threshold space
    tsamples = np.linspace(0, len(T) - 1, n_samples).astype(int)

    # Compute FPR and TPR at each sampled threshold for all folds
    n_folds = len(all_fpr)
    fpr_matrix = np.zeros((n_samples, n_folds))
    tpr_matrix = np.zeros((n_samples, n_folds))
    sampled_thresholds = []

    for n_sample, tidx in enumerate(tsamples):
        sample_threshold = T[tidx]
        sampled_thresholds.append(sample_threshold)

        for i_fold in range(n_folds):
            idx = 0
            while (idx < len(all_thresholds[i_fold]) - 1 and
                   all_thresholds[i_fold][idx] > sample_threshold):
                idx += 1

            fpr_matrix[n_sample, i_fold] = all_fpr[i_fold][idx]
            tpr_matrix[n_sample, i_fold] = all_tpr[i_fold][idx]

    # Compute confidence intervals for FPR and TPR at each sampled threshold
    roc_data = []
    for n_sample in range(n_samples):
        fpr_values = fpr_matrix[n_sample, :]
        tpr_values = tpr_matrix[n_sample, :]

        fpr_ci = compute_confidence(fpr_values, median_n_train, median_n_test, alpha)
        tpr_ci = compute_confidence(tpr_values, median_n_train, median_n_test, alpha)

        roc_data.append({
            'threshold': sampled_thresholds[n_sample],
            'FPR': f"[{fpr_ci[0]:.8f} {fpr_ci[1]:.8f}]",
            'TPR': f"[{tpr_ci[0]:.8f} {tpr_ci[1]:.8f}]"
        })

    return pd.DataFrame(roc_data)


def main(args):

    res_path = Path(args.res_path)
    modalities = args.modalities
    alpha = args.alpha
    n_samples = args.n_samples

    for modality in modalities:
        print(f"\nProcessing modality: {modality}")

        all_metrics = []
        for fold in range(20):
            fold_num = str(fold)
            result = extract_from_one_fold(res_path, modality, fold_num)
            if result is not None:
                all_metrics.append(result)

        if not all_metrics:
            print(f"No valid fold data found for {modality}. Skipping.")
            continue

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df[['auroc', 'accuracy', 'sensitivity', 'specificity',
                                  'n_samples', 'n_positive', 'n_negative',
                                  'tp', 'tn', 'fp', 'fn', 'fold']]

        # Save per-fold metrics CSV
        res_save_path = '/projects/prjs1779/Osteosarcoma/OS_ViT_res'
        out_dir = res_save_path / modality
        os.makedirs(out_dir, exist_ok=True)
        metrics_output = out_dir / "metrics_CI.csv"
        metrics_df.round(2).to_csv(metrics_output, index=False)
        print(f"Fold metrics saved to: {metrics_output}")

        # Compute CIs for each metric
        auc_values = metrics_df['auroc'].dropna().values
        accuracy_values = metrics_df['accuracy'].dropna().values
        sensitivity_values = metrics_df['sensitivity'].dropna().values
        specificity_values = metrics_df['specificity'].dropna().values

        distribution_df = pd.read_csv(LABEL_DIS)
        median_n_train = int(np.mean(distribution_df[distribution_df['modality'] == modality]['train_total'].values))
        median_n_test = int(np.mean(distribution_df[distribution_df['modality'] == modality]['test_total'].values))

        print(f"\nSample sizes: train={median_n_train}, test={median_n_test}")
        print(f"\nConfidence Intervals ({alpha*100:.0f}%):")
        print("-" * 50)

        for name, values in [('AUC', auc_values), ('Accuracy', accuracy_values),
                              ('Sensitivity', sensitivity_values), ('Specificity', specificity_values)]:
            if len(values) > 0:
                ci = compute_confidence(values, median_n_train, median_n_test, alpha)
                print(f"{name}: {np.mean(values):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={len(values)} folds)")
            else:
                print(f"{name}: No valid values")

        # Generate ROC curve data with confidence intervals
        print(f"\nGenerating ROC curve data with confidence intervals...")
        roc_df = generate_roc_with_ci(res_path, modality, alpha, n_samples)

        if roc_df is not None:
            roc_output = out_dir / "roc_curve_ci.csv"
            roc_df.to_csv(roc_output, index=False)
            print(f"ROC curve data saved to: {roc_output}")
        else:
            print("Failed to generate ROC curve data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ViT results and save to CSV.")
    parser.add_argument("res_path", default="/scratch-shared/xwan1/runs_optuna_OS", help="Path to all ViT results.")
    parser.add_argument("modalities", nargs="+", default=["T1W"], help="Modalities to process (e.g., T1W T2W_FS T1W_FS_C).")
    parser.add_argument("--alpha", type=float, default=0.95, help="Confidence level for CI.")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of sample points for ROC curve.")

    args = parser.parse_args()

    main(args)
