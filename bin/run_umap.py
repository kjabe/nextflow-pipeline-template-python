import sys
import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def run_umap(input_file, outdir):
    # Load the dataset with error handling
    try:
        print(f"Reading input file: {input_file}")
        all_dataset = pd.read_pickle(input_file)
        print(f"Dataset loaded successfully. Shape: {all_dataset.shape}")
        print(f"Columns: {all_dataset.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

    try:
        # Filter and transform data for UMAP
        print("Filtering Assay Controls...")
        all_dataset_filtered = all_dataset[all_dataset['Group'] != 'Assay Control'].copy()
        print(f"Filtered dataset shape: {all_dataset_filtered.shape}")

        # Transform to wide format
        print("Transforming to wide format...")
        all_dataset_wide = all_dataset_filtered.pivot_table(
            index=['ID', 'Batch', 'Group'],
            columns='targetName',
            values='Value'
        ).reset_index()
        print(f"Wide format shape: {all_dataset_wide.shape}")

        # Prepare data for UMAP
        X = all_dataset_wide.drop(columns=['ID', 'Batch', 'Group']).values
        print(f"UMAP input shape: {X.shape}")

        # Run UMAP
        print("Running UMAP...")
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            metric='euclidean'
        )
        umap_result = umap_model.fit_transform(X)
        print("UMAP transformation completed")

        # Create DataFrame with UMAP results
        umap_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'ID': all_dataset_wide['ID'],
            'Batch': all_dataset_wide['Batch'],
            'Group': all_dataset_wide['Group']
        })
        print("UMAP DataFrame created")

        # Save plots to output directory
        batch_plot_path = os.path.join(outdir, 'umap_batch.pdf')
        group_plot_path = os.path.join(outdir, 'umap_group.pdf')

        # Create plots with error handling
        try:
            # Plot by Batch
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=umap_df,
                x='UMAP1',
                y='UMAP2',
                hue='Batch',
                palette='deep',
                s=50,
                alpha=0.7
            )
            plt.title('UMAP Projection by Batch')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(batch_plot_path, bbox_inches='tight')

            plt.close()
            print("Batch plot saved as umap_batch.pdf")

            # Plot by Group
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=umap_df,
                x='UMAP1',
                y='UMAP2',
                hue='Group',
                palette='deep',
                s=50,
                alpha=0.7
            )
            plt.title('UMAP Projection by Group')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(group_plot_path, bbox_inches='tight')
            plt.close()
            print("Group plot saved as umap_group.pdf")
        
        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            raise

        # Verify file creation
        output_files = ['umap_batch.pdf', 'umap_group.pdf']
        for file in output_files:
            if os.path.exists(file):
                print(f"Verified: {file} exists ({os.path.getsize(file)} bytes)")
            else:
                print(f"Warning: {file} was not created")

    except Exception as e:
        print(f"Error in UMAP processing: {str(e)}")
        print("\nDebugging information:")
        print(f"Working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        if 'all_dataset_filtered' in locals():
            print(f"Filtered dataset info:")
            print(all_dataset_filtered.info())
        if 'all_dataset_wide' in locals():
            print(f"Wide format dataset info:")
            print(all_dataset_wide.info())
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Error: Expected 2 argument, got {len(sys.argv)-1}")
        print("Usage: python run_umap.py <input_file> <outdir>")
        sys.exit(1)

    input_file = sys.argv[1]
    outdir = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    run_umap(input_file, outdir)
