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

def preprocess_data(df, target_column='Target'):
    """
    Melakukan preprocessing dasar pada dataset (tanpa split dan standarisasi)
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
    
    # === FEATURE ENGINEERING ===
    print("\n4. Feature Engineering:")
    try:
        df_processed['Power'] = df_processed['Torque [Nm]'] * df_processed['Rotational speed [rpm]']
        df_processed['Temp_Diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
        df_processed['Wear_Strain'] = df_processed['Tool wear [min]'] * df_processed['Torque [Nm]']
        print("   ✓ Fitur Power, Temp_Diff, Wear_Strain berhasil ditambahkan.")
    except Exception as e:
        print(f"   ✗ Error saat menambahkan fitur: {e}")
    # ===========================
    
    # Encoding categorical columns
    print("\n5. Encoding data kategorikal:")
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
        print(f"\n6. Kolom yang dihapus: {columns_to_drop}")
    
    # Handle outliers
    print("\n7. Handling outlier (metode IQR):")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    df_processed, outlier_counts = handle_outliers_iqr(df_processed, numeric_cols, method='cap')
    print(f"   Jumlah outlier per kolom:")
    for col, count in outlier_counts.items():
        print(f"   - {col}: {count} outliers")
    
    print("\n" + "="*50)
    print("PREPROCESSING SELESAI")
    print("="*50)
    print(f"Dataset final shape: {df_processed.shape}")
    
    return df_processed, label_encoders

def save_preprocessed_data(df_processed, label_encoders, output_dir='../Eksperimen_SML_Zayga'):
    """
    Menyimpan data yang sudah diproses (preprocessing saja)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed dataframe
    df_processed.to_csv(f'{output_dir}/preprocessed_data.csv', index=False)
    
    # Save label encoders
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    
    # Create summary file
    summary = {
        'total_samples': len(df_processed),
        'total_features': len(df_processed.columns),
        'feature_names': list(df_processed.columns),
        'preprocessing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    pd.DataFrame([summary]).to_csv(f'{output_dir}/preprocessing_summary.csv', index=False)
    
    print(f"\n✓ Data tersimpan di: {output_dir}")
    print(f"Files: {os.listdir(output_dir)}")

def run_preprocessing_pipeline(filepath, target_column='Target'):
    """
    Menjalankan pipeline preprocessing (tanpa split dan standarisasi)
    """
    # Load data
    df = load_data(filepath)
    
    if df is None:
        return None
    
    # Preprocessing only
    df_processed, label_encoders = preprocess_data(df, target_column)
    
    # Save preprocessed data
    save_preprocessed_data(df_processed, label_encoders)
    
    return df_processed, label_encoders

if __name__ == "__main__":
    # Determine filepath based on environment
    if os.path.exists('../predictive_maintenance.raw'):
        filepath = '../predictive_maintenance.raw'
    elif os.path.exists('predictive_maintenance.raw'):
        filepath = 'predictive_maintenance.raw'
    else:
        print("Dataset file not found!")
        exit(1)
    
    # Run preprocessing pipeline
    result = run_preprocessing_pipeline(
        filepath=filepath,
        target_column='Target'
    )
    
    if result is not None:
        df_processed, label_encoders = result
        print(f"\n✓ Preprocessing pipeline completed successfully!")
        print(f"Final dataset shape: {df_processed.shape}")
        print(f"Columns: {list(df_processed.columns)}")
    else:
        print("\n✗ Preprocessing pipeline failed!")
        exit(1)