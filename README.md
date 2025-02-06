# Nextflow Pipeline Template for Python

This repository contains a template for a Nextflow pipeline designed to process and analyze data using Python scripts. The pipeline includes data loading, UMAP dimensionality reduction, and Elastic Net modeling.

## Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/kjabe/nextflow-pipeline-template-python.git
   cd nextflow-pipeline-template-python
   ```

2. **Install Nextflow:**

   Follow the instructions on the [Nextflow website](https://www.nextflow.io/) to install Nextflow.

3. **Create and activate the Conda environment:**

   ```sh
   conda env create -f env.yaml
   conda activate nextflow_pipeline_python
   ```

## Running the Pipeline

1. **Prepare the input data:**

   Place the input data files in the `data` directory. The expected files are:

   - `sample_id.xlsx`
   - `data1.xlsx`
   - `data2.xlsx`

2. **Configure the pipeline:**

   Edit the `nextflow.config` file to set the correct paths for the input data and output directory.

3. **Run the pipeline:**

   ```sh
   nextflow run main.nf
   ```

## Python Scripts

- `load_data.py`: This script loads and preprocesses the input data files, merging them into a single dataset.

- `run_umap.py`: This script performs UMAP dimensionality reduction on the preprocessed data and generates plots.

- `elastic_net.py`: This script trains an Elastic Net model on the preprocessed data, generates ROC curves, and saves the model and coefficients.

## Output

The pipeline generates various output files in the `analysis` directory, including:

- UMAP plots (`umap_batch.pdf`, `umap_group.pdf`)
- Elastic Net model files (`final_model.joblib`, `auc_value.joblib`, `model_coefficients.joblib`)
- ROC curve plot (`roc_curve.pdf`)
- Coefficients plot (`coefficients_plot.pdf`)
- Batch distribution plots (`train_batch_distribution.pdf`, `test_batch_distribution.pdf`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.