import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Memuat dataset dari file CSV
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset berhasil dimuat dengan {len(df)} baris dan {len(df.columns)} kolom")
        return df
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return None

def handle_outliers_iqr(df, columns, method='cap'):
    """
    Menangani outlier menggunakan metode IQR
    """
    df_clean = df.copy()
    outlier_info = {}
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_info[col] = len(outliers)
        
        if method == 'cap':
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean, outlier_info

def preprocess_data(df, target_column='Target', test_size=0.2, random_state=42):
    """
    Melakukan preprocessing lengkap pada dataset
    """
    print("="*50)
    print("MEMULAI PREPROCESSING DATA")
    print("="*50)
    
    df_processed = df.copy()
    print(f"\n1. Jumlah data awal: {len(df_processed)}")
    
    # Handle missing values
    missing_values = df_processed.isnull().sum().sum()
    print(f"\n2. Missing values: {missing_values}")
    if missing_values > 0:
        df_processed = df_processed.dropna()
        print(f"   Data setelah menghapus missing values: {len(df_processed)}")
    
    # Handle duplicates
    duplicates = df_processed.duplicated().sum()
    print(f"\n3. Duplikat data: {duplicates}")
    if duplicates > 0:
        df_processed = df_processed.drop_duplicates()
        print(f"   Data setelah menghapus duplikat: {len(df_processed)}")
    
    # Encoding categorical columns
    print("\n4. Encoding data kategorikal:")
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    for col in categorical_cols:
        if col not in ['UDI', 'Product ID']:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            print(f"   - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Drop unnecessary columns
    columns_to_drop = []
    for col in ['UDI', 'Product ID']:
        if col in df_processed.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_processed = df_processed.drop(columns=columns_to_drop)
        print(f"\n5. Kolom yang dihapus: {columns_to_drop}")
    
    # Handle outliers
    print("\n6. Handling outlier (metode IQR):")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    df_processed, outlier_counts = handle_outliers_iqr(df_processed, numeric_cols, method='cap')
    print(f"   Jumlah outlier per kolom:")
    for col, count in outlier_counts.items():
        print(f"   - {col}: {count} outliers")
    
    # Separate features and target
    print(f"\n7. Memisahkan fitur dan target:")
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    print(f"   Shape fitur (X): {X.shape}")
    print(f"   Shape target (y): {y.shape}")
    print(f"   Distribusi kelas target:\n{y.value_counts()}")
    
    # Split data
    print(f"\n8. Split data (train: {1-test_size:.0%}, test: {test_size:.0%}):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Standardize features
    print("\n9. Standarisasi fitur numerik:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Standarisasi selesai")
    
    print("\n" + "="*50)
    print("PREPROCESSING SELESAI")
    print("="*50)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

def save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoders, output_dir='../data/processed'):
    """
    Menyimpan data yang sudah diproses
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    # Save scaler and encoders
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    
    # Create summary file
    summary = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'preprocessing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    pd.DataFrame([summary]).to_csv(f'{output_dir}/preprocessing_summary.csv', index=False)
    
    print(f"\n✓ Data tersimpan di: {output_dir}")
    print(f"Files: {os.listdir(output_dir)}")

def run_preprocessing_pipeline(filepath, target_column='Target', test_size=0.2, random_state=42):
    """
    Menjalankan pipeline preprocessing lengkap
    """
    # Load data
    df = load_data(filepath)
    
    if df is None:
        return None
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(
        df, target_column, test_size, random_state
    )
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoders)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    # Determine filepath based on environment
    if os.path.exists('../data/raw/predictive_maintenance.raw'):
        filepath = '../data/raw/predictive_maintenance.raw'
    elif os.path.exists('predictive_maintenance.raw'):
        filepath = 'predictive_maintenance.raw'
    else:
        print("Dataset file not found!")
        exit(1)
    
    # Run preprocessing pipeline
    result = run_preprocessing_pipeline(
        filepath=filepath,
        target_column='Target',
        test_size=0.2,
        random_state=42
    )
    
    if result is not None:
        print("\n✓ Preprocessing pipeline completed successfully!")
    else:
        print("\n✗ Preprocessing pipeline failed!")
        exit(1)