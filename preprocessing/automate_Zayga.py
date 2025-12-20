import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Memuat dataset dari file CSV
    
    Parameters:
    filepath (str): Path ke file dataset
    
    Returns:
    pd.DataFrame: Dataset yang telah dimuat
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
    
    Parameters:
    df (pd.DataFrame): Dataframe yang akan diproses
    columns (list): List kolom yang akan dihandle outliernya
    method (str): 'cap' untuk capping, 'remove' untuk menghapus outlier
    
    Returns:
    pd.DataFrame: Dataframe setelah handling outlier
    dict: Informasi jumlah outlier per kolom
    """
    df_clean = df.copy()
    outlier_info = {}
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Hitung jumlah outlier
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_info[col] = len(outliers)
        
        if method == 'cap':
            # Capping: ganti outlier dengan batas atas/bawah
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # Hapus baris yang mengandung outlier
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean, outlier_info

def preprocess_data(df, target_column='Target', test_size=0.2, random_state=42):
    """
    Melakukan preprocessing lengkap pada dataset
    
    Parameters:
    df (pd.DataFrame): Dataset yang akan diproses
    target_column (str): Nama kolom target
    test_size (float): Proporsi data test (default: 0.2)
    random_state (int): Random state untuk reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler, label_encoders)
    """
    print("="*50)
    print("MEMULAI PREPROCESSING DATA")
    print("="*50)
    
    # 1. Membuat copy dataframe
    df_processed = df.copy()
    print(f"\n1. Jumlah data awal: {len(df_processed)}")
    
    # 2. Cek dan handle missing values
    missing_values = df_processed.isnull().sum().sum()
    print(f"\n2. Missing values: {missing_values}")
    if missing_values > 0:
        df_processed = df_processed.dropna()
        print(f"   Data setelah menghapus missing values: {len(df_processed)}")
    
    # 3. Cek dan handle duplicates
    duplicates = df_processed.duplicated().sum()
    print(f"\n3. Duplikat data: {duplicates}")
    if duplicates > 0:
        df_processed = df_processed.drop_duplicates()
        print(f"   Data setelah menghapus duplikat: {len(df_processed)}")
    
    # 4. Encoding kolom kategorikal
    print("\n4. Encoding data kategorikal:")
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Hapus target column dari categorical cols jika ada
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    for col in categorical_cols:
        if col not in ['UDI', 'Product ID']:  # Skip ID columns
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            print(f"   - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 5. Hapus kolom yang tidak diperlukan
    columns_to_drop = []
    for col in ['UDI', 'Product ID']:
        if col in df_processed.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_processed = df_processed.drop(columns=columns_to_drop)
        print(f"\n5. Kolom yang dihapus: {columns_to_drop}")
    
    # 6. Handling outlier
    print("\n6. Handling outlier (metode IQR):")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    df_processed, outlier_counts = handle_outliers_iqr(df_processed, numeric_cols, method='cap')
    print(f"   Jumlah outlier per kolom:")
    for col, count in outlier_counts.items():
        print(f"   - {col}: {count} outliers")
    
    # 7. Pisahkan fitur dan target
    print(f"\n7. Memisahkan fitur dan target:")
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    print(f"   Shape fitur (X): {X.shape}")
    print(f"   Shape target (y): {y.shape}")
    print(f"   Distribusi kelas target:\n{y.value_counts()}")
    
    # 8. Split data
    print(f"\n8. Split data (train: {1-test_size:.0%}, test: {test_size:.0%}):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 9. Standarisasi fitur
    print("\n9. Standarisasi fitur numerik:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Standarisasi selesai")
    
    print("\n" + "="*50)
    print("PREPROCESSING SELESAI")
    print("="*50)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

def run_preprocessing_pipeline(filepath, target_column='Target', test_size=0.2, random_state=42):
    """
    Menjalankan pipeline preprocessing lengkap dari file hingga data siap dilatih
    
    Parameters:
    filepath (str): Path ke file dataset
    target_column (str): Nama kolom target
    test_size (float): Proporsi data test
    random_state (int): Random state untuk reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler, label_encoders)
    """
    # Load data
    df = load_data(filepath)
    
    if df is None:
        return None
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(
        df, target_column, test_size, random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

# Contoh penggunaan
if __name__ == "__main__":
    # Path ke dataset
    filepath = r'D:\Eksperimen_SML_Zayga\predictive_maintenance.raw'
    
    # Jalankan preprocessing pipeline
    result = run_preprocessing_pipeline(
        filepath=filepath,
        target_column='Target',
        test_size=0.2,
        random_state=42
    )
    
    if result is not None:
        X_train, X_test, y_train, y_test, scaler, label_encoders = result
        print("\n✓ Data siap untuk dilatih!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")