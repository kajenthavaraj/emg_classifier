# EMG Classifier for Rock Paper Scissors

A machine learning project that uses Electromyography (EMG) signals to classify hand gestures for Rock Paper Scissors game.

## 📋 Project Overview

This project demonstrates how to use EMG signals from forearm muscles to predict hand gestures. The classifier can distinguish between different finger positions that correspond to Rock, Paper, and Scissors hand gestures.

## 🗂️ Project Structure

```
emg_classifier/
├── emg_data/           # EMG signal datasets
│   ├── 0.csv          # Class 0 data
│   ├── 1.csv          # Class 1 data
│   ├── 2.csv          # Class 2 data
│   └── 3.csv          # Class 3 data
├── notebook.ipynb     # Main analysis and training notebook
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Installation section)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd emg_classifier
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow keras
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**
   - Navigate to `notebook.ipynb` in the Jupyter interface
   - Run cells sequentially to load, preprocess, and train the model

## 📊 Data Description

### EMG Data Format
- **64 channels**: EMG signals from different electrode positions
- **4 classes**: Different finger positions (0, 1, 2, 3)
- **Features**: Raw EMG signal values
- **Target**: Finger position labels

### Data Preprocessing
The notebook includes:
- Data loading from CSV files
- Column standardization (renaming to 0-63)
- Feature scaling using StandardScaler
- Train-test split for model evaluation

## 🔧 Usage

### Running the Analysis

1. **Load the data**:
   ```python
   import pandas as pd
   
   dfs = []
   for label in ['0', '1', '2', '3']:
       dfs.append(pd.read_csv('emg_data/' + label + '.csv'))
   ```

2. **Preprocess the data**:
   ```python
   # Standardize column names
   for df in dfs:
       df.columns = list(range(len(df.columns)))
   
   # Combine all datasets
   combined_df = pd.concat(dfs, ignore_index=True)
   ```

3. **Extract features and labels**:
   ```python
   # EMG values (columns 0-63)
   emg_values = combined_df.iloc[:, 0:64]
   
   # Finger position labels (column 64)
   finger_position = combined_df[64].copy()
   ```

4. **Scale the features**:
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   emg_values = scaler.fit_transform(emg_values)
   ```

## 🧠 Machine Learning Pipeline

The project implements a complete ML pipeline:

1. **Data Loading**: Read EMG data from CSV files
2. **Data Preprocessing**: Clean and standardize the data
3. **Feature Engineering**: Extract relevant features from EMG signals
4. **Model Training**: Train classification models
5. **Evaluation**: Assess model performance with metrics
6. **Prediction**: Use trained model for new EMG signal classification

## 📈 Expected Results

- **Classification accuracy**: Varies based on data quality and model choice
- **Classes**: 4 different finger positions
- **Features**: 64 EMG channels per sample

## 🔬 Technical Details

### EMG Signal Processing
- **Sampling**: EMG signals are typically sampled at high frequencies
- **Filtering**: Noise reduction and signal conditioning
- **Feature Extraction**: Time-domain and frequency-domain features

### Machine Learning Models
- Traditional ML: Random Forest, SVM, etc.
- Deep Learning: Neural networks with TensorFlow/Keras
- Ensemble Methods: Combining multiple models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- EMG data collection and preprocessing
- Machine learning model development
- Signal processing techniques

## 📞 Contact

For questions or contributions, please open an issue in the repository.

---

**Note**: This is a research/educational project. For medical applications, consult with healthcare professionals and ensure proper validation protocols. 