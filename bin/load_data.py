import sys
import pandas as pd
import os

def load_data(id_path, data1_path, data2_path, outdir):
    print(f"Reading files from:")
    print(f"ID path: {id_path}")
    print(f"data1 path: {data1_path}")
    print(f"data2 path: {data2_path}")
    
    # Read the data
    id_data = pd.read_excel(id_path)
    print("Original ID data columns:", id_data.columns.tolist())
    
    # Clean column names (remove trailing spaces)
    id_data.columns = id_data.columns.str.strip()
    print("Cleaned ID data columns:", id_data.columns.tolist())
    
    # Ensure correct column names
    if 'Group' not in id_data.columns and 'Unnamed: 1' in id_data.columns:
        id_data = id_data.rename(columns={'Unnamed: 1': 'Group'})
    
    if 'ID' not in id_data.columns and 'ID ' in id_data.columns:
        id_data = id_data.rename(columns={'ID ': 'ID'})
    
    print("Final ID data columns:", id_data.columns.tolist())

    data1_data = pd.read_excel(data1_path)
    data2_data = pd.read_excel(data2_path)

    # Reshape the data into long format and add Batch information
    long_data1 = (data1_data.melt(id_vars=['targetName'], var_name='ID', value_name='Value')
                    .assign(Batch='Plate1'))

    long_data2 = (data2_data.melt(id_vars=['targetName'], var_name='ID', value_name='Value')
                    .assign(Batch='Plate2'))

    # Combine the datasets
    long_data = pd.concat([long_data1, long_data2], ignore_index=True)
    print("Combined data shape:", long_data.shape)
    print("data columns:", long_data.columns.tolist())
    print("Sample of ID values in data:", long_data['ID'].head().tolist())
    print("Sample of ID values in id_data:", id_data['ID'].head().tolist())

    try:
        # Join the IDs with the group information and organize columns
        all_dataset = (long_data.merge(id_data, how='left', on='ID')
                      .loc[:, ['ID', 'Group', 'targetName', 'Value', 'Batch']]
                      .sort_values(by=['ID', 'targetName']))

        # Fill missing Group values with 'Assay Control'
        all_dataset['Group'] = all_dataset['Group'].fillna('Assay Control')
        
        print("Final dataset shape:", all_dataset.shape)
        print("Final dataset columns:", all_dataset.columns.tolist())
        print("Sample of final dataset:")
        print(all_dataset.head())

        # Save the dataset to a pickle file
        all_dataset.to_pickle('all_dataset.pkl')
        print("Dataset saved successfully to all_dataset.pkl")
        
        # Verify the file was created
        if os.path.exists('all_dataset.pkl'):
            print("Verified: all_dataset.pkl exists")
            print("File size:", os.path.getsize('all_dataset.pkl'), "bytes")
        else:
            print("Warning: all_dataset.pkl was not created")

    except Exception as e:
        print(f"Error during merge operation: {str(e)}")
        print("\nDebugging information:")
        print(f"long_data shape: {long_data.shape}")
        print(f"id_data shape: {id_data.shape}")
        print(f"long_data columns: {long_data.columns.tolist()}")
        print(f"id_data columns: {id_data.columns.tolist()}")
        print("\nFirst few rows of each DataFrame:")
        print("\nlong_data head:")
        print(long_data.head())
        print("\nid_data head:")
        print(id_data.head())
        raise

    # Save the dataset to the output directory
    output_path = os.path.join(outdir, 'all_dataset.pkl')
    all_dataset.to_pickle(output_path)
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Error: Expected 4 arguments, got {len(sys.argv)-1}")
        print("Usage: python load_data.py <id_path> <data1_path> <data2_path> <outdir>")

        sys.exit(1)
        
    id_path = sys.argv[1]
    data1_path = sys.argv[2]
    data2_path = sys.argv[3]
    outdir = sys.argv[4]
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    if not all(os.path.exists(f) for f in [id_path, data1_path, data2_path]):
        missing = [f for f in [id_path, data1_path, data2_path] if not os.path.exists(f)]
        print(f"Error: Missing files: {missing}")
        sys.exit(1)
        
    load_data(id_path, data1_path, data2_path, outdir)
