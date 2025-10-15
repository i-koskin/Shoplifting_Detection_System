import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import RandomizedSearchCV
import time
import matplotlib.pyplot as plt
import warnings
import logging
import sys
import os
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xgboost_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings('ignore')


class XGBoostOptimizer:
    """
    A comprehensive XGBoost model optimizer with hyperparameter tuning and evaluation.

    This class provides methods for sequential hyperparameter optimization, 
    model evaluation, cross-validation, and result visualization.
    """

    def __init__(self, random_state=42):
        """
        Initialize the XGBoost optimizer.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.optimization_history = []
        logger.info(
            f"XGBoostOptimizer initialized with random_state={random_state}")

    def sequential_optimization(self, X_train, y_train, X_val, y_val):
        """
        Enhanced sequential optimization of hyperparameters by parameter groups.

        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_val (array-like): Validation features
            y_val (array-like): Validation labels

        Returns:
            tuple: (best_params, best_score) containing optimized parameters and best score

        Raises:
            Exception: If optimization fails at any phase
        """
        best_score = 0
        best_params = {}

        logger.info("Starting sequential hyperparameter optimization...")

        try:
            # Phase 1: Core architectural parameters
            logger.info("Phase 1: Optimizing core parameters...")
            phase1_params = {
                'max_depth': randint(3, 12),
                'min_child_weight': randint(1, 10),
                'gamma': uniform(0, 0.5)
            }

            model_phase1 = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic',
                tree_method='hist',
                enable_categorical=True
            )

            search1 = RandomizedSearchCV(
                model_phase1, phase1_params, n_iter=15,
                cv=StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=self.random_state),
                scoring='roc_auc', random_state=self.random_state, n_jobs=-1,
                verbose=1
            )
            search1.fit(X_train, y_train, eval_set=[
                        (X_val, y_val)], verbose=False)

            best_params.update(search1.best_params_)
            best_score = search1.best_score_
            logger.info(f"Best score after phase 1: {best_score:.4f}")
            self.optimization_history.append(
                ('Phase 1', best_score, best_params.copy()))

            # Phase 2: Sampling and regularization parameters
            logger.info("Phase 2: Optimizing sampling and regularization...")
            phase2_params = {
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': loguniform(1e-5, 10),
                'reg_lambda': loguniform(1e-5, 10)
            }

            model_phase2 = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                **best_params,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic',
                tree_method='hist',
                enable_categorical=True
            )

            search2 = RandomizedSearchCV(
                model_phase2, phase2_params, n_iter=15,
                cv=StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=self.random_state),
                scoring='roc_auc', random_state=self.random_state, n_jobs=-1,
                verbose=1
            )
            search2.fit(X_train, y_train, eval_set=[
                        (X_val, y_val)], verbose=False)

            best_params.update(search2.best_params_)
            best_score = search2.best_score_
            logger.info(f"Best score after phase 2: {best_score:.4f}")
            self.optimization_history.append(
                ('Phase 2', best_score, best_params.copy()))

            # Phase 3: Final fine-tuning
            logger.info("Phase 3: Final fine-tuning...")
            phase3_params = {
                'learning_rate': loguniform(1e-3, 0.3),
                'n_estimators': randint(100, 1000)
            }

            model_phase3 = xgb.XGBClassifier(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic',
                tree_method='hist',
                enable_categorical=True
            )

            search3 = RandomizedSearchCV(
                model_phase3, phase3_params, n_iter=10,
                cv=StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=self.random_state),
                scoring='roc_auc', random_state=self.random_state, n_jobs=-1,
                verbose=1
            )
            search3.fit(X_train, y_train, eval_set=[
                        (X_val, y_val)], verbose=False)

            best_params.update(search3.best_params_)
            best_score = search3.best_score_
            logger.info(f"Best score after phase 3: {best_score:.4f}")
            self.optimization_history.append(
                ('Phase 3', best_score, best_params.copy()))

            return best_params, best_score

        except Exception as e:
            logger.error(f"Sequential optimization failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def create_final_model(self, optimized_params, X_train, y_train, X_val, y_val):
        """
        Create and train final model with optimized parameters.

        Args:
            optimized_params (dict): Optimized hyperparameters
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_val (array-like): Validation features
            y_val (array-like): Validation labels

        Returns:
            tuple: (final_model, final_params) containing trained model and parameters
        """
        logger.info("Creating final model with optimized parameters...")

        try:
            # Base parameters that should always be present
            base_params = {
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 50,
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'enable_categorical': True
            }

            # Merge optimized parameters with base parameters
            final_params = {**optimized_params, **base_params}

            # Ensure n_estimators is sufficient for early stopping
            if final_params.get('n_estimators', 0) < 100:
                final_params['n_estimators'] = 1000

            logger.info("Final model parameters:")
            for key, value in final_params.items():
                logger.info(f"  {key}: {value}")

            # Create and train the model
            final_model = xgb.XGBClassifier(**final_params)

            logger.info("Training final model...")
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )

            return final_model, final_params

        except Exception as e:
            logger.error(f"Final model creation failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def evaluate_model(self, model, X_test, y_test, model_name="Final Model"):
        """
        Comprehensive model evaluation with multiple metrics.

        Args:
            model: Trained model object
            X_test (array-like): Test features
            y_test (array-like): Test labels
            model_name (str): Name of the model for reporting

        Returns:
            tuple: (accuracy, auc_score) containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            logger.info(
                f"{model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {auc_score:.4f}")

            # Detailed classification report
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info("Confusion Matrix:")
            logger.info(f"\n{cm}")

            return accuracy, auc_score

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def manual_cross_validation(self, model_params, X, y, n_splits=5):
        """
        Manual cross-validation without early stopping.

        Args:
            model_params (dict): Model parameters
            X (array-like): Features
            y (array-like): Labels
            n_splits (int): Number of cross-validation folds

        Returns:
            tuple: (auc_scores, accuracy_scores) containing cross-validation results
        """
        logger.info(f"Performing manual {n_splits}-fold cross-validation...")

        try:
            kf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state)
            auc_scores = []
            accuracy_scores = []

            # Remove early_stopping_rounds for cross-validation
            cv_params = model_params.copy()
            if 'early_stopping_rounds' in cv_params:
                del cv_params['early_stopping_rounds']

            # Convert to numpy arrays if pandas objects
            if hasattr(X, 'iloc'):
                X_array = X.values
            else:
                X_array = X

            if hasattr(y, 'iloc'):
                y_array = y.values
            else:
                y_array = y

            # Use tqdm for progress visualization
            for fold, (train_idx, val_idx) in enumerate(
                tqdm(kf.split(X_array, y_array), total=n_splits,
                     desc="Cross-validation folds")
            ):
                logger.info(f"Processing fold {fold + 1}/{n_splits}...")

                # Extract fold data
                X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
                y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]

                # Create and train model for fold
                model = xgb.XGBClassifier(**cv_params)
                model.fit(X_train_fold, y_train_fold, verbose=False)

                # Predictions and evaluation
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                y_pred = model.predict(X_val_fold)

                fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
                fold_accuracy = accuracy_score(y_val_fold, y_pred)

                auc_scores.append(fold_auc)
                accuracy_scores.append(fold_accuracy)

                logger.info(
                    f"Fold {fold + 1}: AUC = {fold_auc:.4f}, Accuracy = {fold_accuracy:.4f}")

            # Log cross-validation results
            logger.info("Cross-validation results:")
            logger.info(
                f"ROC-AUC across folds: {[f'{score:.4f}' for score in auc_scores]}")
            logger.info(
                f"Accuracy across folds: {[f'{score:.4f}' for score in accuracy_scores]}")
            logger.info(
                f"Mean ROC-AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores) * 2:.4f})")
            logger.info(
                f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores) * 2:.4f})")

            return auc_scores, accuracy_scores

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def plot_cross_validation_results(self, cv_results, model_name="XGBoost"):
        """
        Visualize cross-validation results.

        Args:
            cv_results (dict): Cross-validation results dictionary
            model_name (str): Name of the model for plot titles
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # ROC-AUC plot
            folds = range(1, len(cv_results['cv_auc_scores']) + 1)
            ax1.plot(folds, cv_results['cv_auc_scores'],
                     marker='o', linewidth=2, markersize=8)
            ax1.axhline(y=cv_results['cv_auc_mean'], color='r', linestyle='--',
                        label=f'Mean: {cv_results["cv_auc_mean"]:.4f}')
            ax1.fill_between(folds,
                             cv_results['cv_auc_mean'] -
                             cv_results['cv_auc_std'],
                             cv_results['cv_auc_mean'] +
                             cv_results['cv_auc_std'],
                             alpha=0.2, color='gray', label='±1 std')
            ax1.set_xlabel('Fold Number')
            ax1.set_ylabel('ROC-AUC')
            ax1.set_title(f'ROC-AUC by Fold ({model_name})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Accuracy plot
            ax2.plot(folds, cv_results['cv_accuracy_scores'],
                     marker='o', linewidth=2, markersize=8, color='green')
            ax2.axhline(y=cv_results['cv_accuracy_mean'], color='r', linestyle='--',
                        label=f'Mean: {cv_results["cv_accuracy_mean"]:.4f}')
            ax2.fill_between(folds,
                             cv_results['cv_accuracy_mean'] -
                             cv_results['cv_accuracy_std'],
                             cv_results['cv_accuracy_mean'] +
                             cv_results['cv_accuracy_std'],
                             alpha=0.2, color='gray', label='±1 std')
            ax2.set_xlabel('Fold Number')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'Accuracy by Fold ({model_name})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('cross_validation_results.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(
                "Cross-validation results plot saved as 'cross_validation_results.png'")

        except Exception as e:
            logger.error(f"Plotting cross-validation results failed: {e}")
            logger.error(traceback.format_exc())

    def compare_with_baseline(self, optimized_model, X_train, y_train, X_test, y_test):
        """
        Compare optimized model with baseline model.

        Args:
            optimized_model: Optimized model object
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_test (array-like): Test features
            y_test (array-like): Test labels
        """
        logger.info("Comparing with baseline model...")

        try:
            # Baseline model
            baseline_model = xgb.XGBClassifier(
                random_state=self.random_state, n_jobs=-1)
            baseline_model.fit(X_train, y_train, verbose=False)

            # Evaluate baseline model
            baseline_accuracy, baseline_auc = self.evaluate_model(
                baseline_model, X_test, y_test, "Baseline Model"
            )

            # Evaluate optimized model
            optimized_accuracy, optimized_auc = self.evaluate_model(
                optimized_model, X_test, y_test, "Optimized Model"
            )

            # Comparison
            logger.info("Model comparison results:")
            improvement_accuracy = optimized_accuracy - baseline_accuracy
            improvement_auc = optimized_auc - baseline_auc

            improvement_pct_accuracy = (
                improvement_accuracy / baseline_accuracy) * 100
            improvement_pct_auc = (improvement_auc / baseline_auc) * 100

            logger.info(
                f"Accuracy improvement: {improvement_accuracy:+.4f} ({improvement_pct_accuracy:+.2f}%)")
            logger.info(
                f"ROC-AUC improvement: {improvement_auc:+.4f} ({improvement_pct_auc:+.2f}%)")

        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            logger.error(traceback.format_exc())

    def feature_importance_analysis(self, model, feature_names=None, top_n=15):
        """
        Analyze and visualize feature importance.

        Args:
            model: Trained model object
            feature_names (list): List of feature names
            top_n (int): Number of top features to display
        """
        logger.info(f"Analyzing top-{top_n} feature importance...")

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                if feature_names is None:
                    feature_names = [
                        f'Feature_{i}' for i in range(len(importances))]

                # Sort by importance
                indices = np.argsort(importances)[::-1]

                logger.info("Feature importance ranking:")
                for i, idx in enumerate(indices[:top_n]):
                    logger.info(
                        f"{i+1:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}")

                # Plot feature importance
                plt.figure(figsize=(12, 8))
                plt.barh(range(top_n), importances[indices[:top_n]][::-1])
                plt.yticks(range(top_n), [feature_names[i]
                           for i in indices[:top_n]][::-1])
                plt.xlabel('Importance')
                plt.title(f'Top-{top_n} Most Important Features')
                plt.tight_layout()
                plt.savefig('feature_importance.png',
                            dpi=300, bbox_inches='tight')
                logger.info(
                    "Feature importance plot saved as 'feature_importance.png'")

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            logger.error(traceback.format_exc())

    def save_model_and_results(self, model, optimized_params, accuracy, auc_score, cv_results=None):
        """
        Save model and results to files.

        Args:
            model: Trained model object
            optimized_params (dict): Optimized parameters
            accuracy (float): Test accuracy
            auc_score (float): Test ROC-AUC score
            cv_results (dict): Cross-validation results
        """
        logger.info("Saving model and results...")

        try:
            # Save model
            model_path = './trained_model.json'
            model.save_model(model_path)
            logger.info(f"Model saved to '{model_path}'")

            # Save parameters and metrics
            params_df = pd.DataFrame([optimized_params])
            params_df['accuracy'] = accuracy
            params_df['auc_score'] = auc_score
            params_df['timestamp'] = pd.Timestamp.now()

            # Add cross-validation results if available
            if cv_results is not None:
                params_df['cv_auc_mean'] = cv_results['cv_auc_mean']
                params_df['cv_auc_std'] = cv_results['cv_auc_std']
                params_df['cv_accuracy_mean'] = cv_results['cv_accuracy_mean']
                params_df['cv_accuracy_std'] = cv_results['cv_accuracy_std']

            params_df.to_csv('optimization_results.csv', index=False)
            logger.info(
                "Parameters and metrics saved to 'optimization_results.csv'")

            # Save detailed cross-validation results
            if cv_results is not None:
                cv_details = {
                    'auc_scores': cv_results['cv_auc_scores'],
                    'accuracy_scores': cv_results['cv_accuracy_scores'],
                    'fold_numbers': list(range(1, len(cv_results['cv_auc_scores']) + 1))
                }
                cv_df = pd.DataFrame(cv_details)
                cv_df.to_csv('cross_validation_details.csv', index=False)
                logger.info(
                    "Cross-validation details saved to 'cross_validation_details.csv'")

            # Save comprehensive report
            with open('model_parameters.txt', 'w') as f:
                f.write("OPTIMIZED XGBOOST PARAMETERS\n")
                f.write("=" * 40 + "\n")
                for key, value in optimized_params.items():
                    f.write(f"{key}: {value}\n")

                f.write(f"\nTEST METRICS:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"ROC-AUC: {auc_score:.4f}\n")

                if cv_results is not None:
                    f.write(f"\nCROSS-VALIDATION RESULTS:\n")
                    f.write(
                        f"Mean ROC-AUC: {cv_results['cv_auc_mean']:.4f} (+/- {cv_results['cv_auc_std'] * 2:.4f})\n")
                    f.write(
                        f"Mean Accuracy: {cv_results['cv_accuracy_mean']:.4f} (+/- {cv_results['cv_accuracy_std'] * 2:.4f})\n")

            logger.info("Detailed report saved to 'model_parameters.txt'")

        except Exception as e:
            logger.error(f"Saving model and results failed: {e}")
            logger.error(traceback.format_exc())


def load_and_prepare_data(file_path):
    """
    Load and prepare dataset for training.

    Args:
        file_path (str): Path to the dataset CSV file

    Returns:
        tuple: (X, y) containing features and labels

    Raises:
        Exception: If data loading or preparation fails
    """
    try:
        logger.info(f"Loading dataset from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(
            f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        # Check required columns
        required_columns = ['label', 'image_name']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Prepare features and target
        X = df.drop(['label', 'image_name'], axis=1)
        y = df['label'].map({'Suspicious': 0, 'Normal': 1})

        logger.info(
            f"Features shape: {X.shape}, Target distribution:\n{y.value_counts()}")

        return X, y

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """
    Main execution function for XGBoost model optimization pipeline.
    """
    try:
        logger.info("Starting XGBoost optimization pipeline...")

        # Initialize optimizer
        optimizer = XGBoostOptimizer(random_state=42)

        # 1. Data preparation
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 50)

        dataset_path = './dataset_path/dataset.csv'
        X, y = load_and_prepare_data(dataset_path)

        # Train/validation/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )

        logger.info(
            f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # 2. Sequential parameter optimization
        logger.info("\nSTEP 2: SEQUENTIAL PARAMETER OPTIMIZATION")
        logger.info("=" * 50)

        start_time = time.time()
        optimized_params, optimization_score = optimizer.sequential_optimization(
            X_train, y_train, X_val, y_val
        )
        optimization_time = time.time() - start_time

        logger.info(
            f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best optimization score: {optimization_score:.4f}")

        # 3. Final model creation
        logger.info("\nSTEP 3: FINAL MODEL CREATION")
        logger.info("=" * 50)

        final_model, final_params = optimizer.create_final_model(
            optimized_params, X_train, y_train, X_val, y_val
        )

        # 4. Model evaluation
        logger.info("\nSTEP 4: MODEL EVALUATION")
        logger.info("=" * 50)

        test_accuracy, test_auc = optimizer.evaluate_model(
            final_model, X_test, y_test)

        # 5. Cross-validation
        logger.info("\nSTEP 5: CROSS-VALIDATION")
        logger.info("=" * 50)

        auc_scores, accuracy_scores = optimizer.manual_cross_validation(
            final_params, X_temp, y_temp, n_splits=5
        )

        cv_results = {
            'cv_auc_mean': np.mean(auc_scores),
            'cv_auc_std': np.std(auc_scores),
            'cv_accuracy_mean': np.mean(accuracy_scores),
            'cv_accuracy_std': np.std(accuracy_scores),
            'cv_auc_scores': auc_scores,
            'cv_accuracy_scores': accuracy_scores
        }

        # 6. Baseline comparison
        logger.info("\nSTEP 6: BASELINE COMPARISON")
        logger.info("=" * 50)

        optimizer.compare_with_baseline(
            final_model, X_train, y_train, X_test, y_test)

        # 7. Feature importance analysis
        logger.info("\nSTEP 7: FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 50)

        optimizer.feature_importance_analysis(
            final_model, feature_names=X.columns.tolist())

        # 8. Save results
        logger.info("\nSTEP 8: SAVING RESULTS")
        logger.info("=" * 50)

        optimizer.save_model_and_results(
            final_model, final_params, test_accuracy, test_auc, cv_results)

        # 9. Visualization
        logger.info("\nSTEP 9: VISUALIZATION")
        logger.info("=" * 50)

        optimizer.plot_cross_validation_results(
            cv_results, "Optimized XGBoost")

        logger.info("\nOPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)

        return final_model, final_params, test_accuracy, test_auc

    except Exception as e:
        logger.error(f"Optimization pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
