import pandas as pd
import os

def load_url_data(data_dir_path):

    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',', header=None)
    url_data.columns = ['text', 'label']
    url_data['label'] = url_data['label'].astype(str)
    
    class_zero = url_data[url_data['label'] == '0']
    class_one = url_data[url_data['label'] == '1']

    if len(class_zero) > len(class_one):
        class_zero_balanced = class_zero.sample(n=len(class_one), random_state=42)
        url_data_balanced = pd.concat([class_zero_balanced, class_one])
        print(f"Balancing strategy: Undersampled class '0' (non-phishing) to match class '1'.")
    elif len(class_one) > len(class_zero):
        class_one_balanced = class_one.sample(n=len(class_zero), random_state=42)
        url_data_balanced = pd.concat([class_zero, class_one_balanced])
        print(f"Balancing strategy: Undersampled class '1' (phishing) to match class '0'.")
    else:
        url_data_balanced = url_data
        print(f"Balancing strategy: Dataset already balanced or no imbalance detected.")

    url_data_balanced = url_data_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    try:
        url_data_balanced['label'] = pd.to_numeric(url_data_balanced['label'], errors='raise')
    except ValueError as e:
        print(f"ERROR during pd.to_numeric: {e}")
        non_numeric = url_data_balanced[pd.to_numeric(url_data_balanced['label'], errors='coerce').isna()]['label'].unique()
        print(f"Non-numeric values found in 'label' column: {non_numeric}")
        raise # Re-raise the error to stop execution

    return url_data_balanced


def main():
    data_dir_path = './data'
    url_data = load_url_data(data_dir_path)

    print(url_data.head())


if __name__ == '__main__':
    main()
