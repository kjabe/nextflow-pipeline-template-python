import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from joblib import dump

def run_elastic_net(input_file, outdir):
    # Read the input dataset
    all_dataset = pd.read_pickle(input_file)

    # Filter and prepare the data
    filtered_data = all_dataset[
        (all_dataset['Group'] != 'Assay Control')  &
        (all_dataset['Group'].isin(['PTSD', 'CPTSD']))
    ]
    filtered_data = filtered_data.copy()  # Create a copy to avoid SettingWithCopyWarning
    filtered_data.loc[:, 'Group'] = filtered_data['Group'].astype('category')

    # Transform data to wide format
    wide_data = filtered_data.pivot_table(index=['ID', 'Batch', 'Group'], columns='targetName', values='Value').reset_index()

    # Set seed for reproducibility
    np.random.seed(123)

    # Split data (stratified by Batch)
    train_data, test_data = train_test_split(wide_data, test_size=0.3, stratify=wide_data['Batch'])

    # Save plots and results to output directory
    train_dist_path = os.path.join(outdir, 'train_batch_distribution.pdf')
    test_dist_path = os.path.join(outdir, 'test_batch_distribution.pdf')
    roc_path = os.path.join(outdir, 'roc_curve.pdf')
    coef_plot_path = os.path.join(outdir, 'coefficients_plot.pdf')
    model_path = os.path.join(outdir, 'final_model.joblib')
    auc_path = os.path.join(outdir, 'auc_value.joblib')
    coef_path = os.path.join(outdir, 'model_coefficients.joblib')

    # Plot batch distribution
    train_plot = sns.countplot(data=train_data, x='Batch')
    train_plot.set_title("Batch Distribution in Training Data")
    plt.tight_layout()
    plt.savefig(train_dist_path)
    plt.clf()
    
    test_plot = sns.countplot(data=test_data, x='Batch')
    test_plot.set_title("Batch Distribution in Testing Data")
    plt.tight_layout()
    plt.savefig(test_dist_path)
    plt.clf()

    # Prepare matrices for training
    X_train = train_data.drop(columns=['ID', 'Group', 'Batch'])
    y_train = train_data['Group'].cat.codes

    # Prepare matrices for testing
    X_test = test_data.drop(columns=['ID', 'Group', 'Batch'])
    y_test = test_data['Group'].cat.codes

    # Normalize predictors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Elastic Net model with cross-validation
    n_repeats = 10
    alpha_value = 0.5
    lambdas = []

    for _ in range(n_repeats):
        cv_fit = ElasticNetCV(l1_ratio=alpha_value, n_alphas=100, cv=5, random_state=123).fit(X_train_scaled, y_train)
        lambdas.append(cv_fit.alpha_)

    # Compute average best lambda
    avg_best_lambda = np.mean(lambdas)

    # Train final model
    final_model = ElasticNetCV(l1_ratio=alpha_value, alphas=[avg_best_lambda], cv=5, random_state=123)
    final_model.fit(X_train_scaled, y_train)

    # Predict on test data
    predictions = final_model.predict(X_test_scaled)

    # Create ROC curve and calculate AUROC
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"AUROC (95% CI) = {roc_auc:.2f}")
    plt.savefig(roc_path)
    plt.clf()

    # Extract and plot coefficients
    coefs = final_model.coef_
    features = X_train.columns.tolist()  # Get feature names
    
    # Create DataFrame with proper column names
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefs
    })
    
    # Filter non-zero coefficients and add category
    coef_df = coef_df[coef_df['Coefficient'] != 0]
    coef_df['Category'] = np.where(coef_df['Coefficient'] < 0, 
                                  'Increased probability of PTSD', 
                                  'Increased probability of CPTSD')

    # Create the barplot with proper column names
    coef_plot = sns.barplot(
        data=coef_df,
        x='Coefficient',
        y='Feature',  # Changed from 'index' to 'Feature'
        hue='Category'
    )
    coef_plot.set_title('Targets Multivariately Discriminating PTSD vs. CPTSD')
    coef_plot.set(xlabel='Elastic Net Coefficient', ylabel='Target')
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(coef_plot_path)
    plt.clf()

    # Save results to output directory
    dump(final_model, model_path)
    dump(roc_auc, auc_path)
    dump(coef_df, coef_path)

    # Printing out file paths to verify
    print(f"Model saved at: {model_path}")
    print(f"AUC value saved at: {auc_path}")
    print(f"Coefficients saved at: {coef_path}")
    print(f"ROC curve saved at: {roc_path}")
    print(f"Coefficients plot saved at: {coef_plot_path}")
    print(f"Training batch distribution plot saved at: {train_dist_path}")
    print(f"Testing batch distribution plot saved at: {test_dist_path}")

    # Check existence of files
    for path in [model_path, auc_path, coef_path, roc_path, coef_plot_path, train_dist_path, test_dist_path]:
        if os.path.isfile(path):
            print(f"File {path} exists and is ready in the output directory.")
        else:
            print(f"File {path} does not exist.")

if __name__ == "__main__":
    input_file = sys.argv[1]
    outdir = sys.argv[2]

    run_elastic_net(input_file, outdir)