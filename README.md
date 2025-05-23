# gtrsb-implementation

1. **Prepare the Data**
   - Ensure you have the dataset folders `train` and `test` in your project directory.
   - Move the training metadata file from the `train` folder to a new folder called `trainFeatures`:
     - `train/train_metadata.csv` → `trainFeatures/train_metadata.csv`
   - Move the test metadata file from the `test` folder to a new folder called `testFeatures`:
     - `test/test_metadata.csv` → `testFeatures/test_metadata.csv`

2. **Install Dependencies**
   - Install the required Python packages using pip:
     ```
     pip install -r requirements.txt
     ```

3. **Preprocess the dataset**
   - Run preprocessing.py to get the relevant files for 

4. **Run the Models**
   - You can run the main script to train and evaluate the models:
     ```
     python main.py
     ```
   - The script will handle training, evaluation, and saving predictions for SVM, KNN, CNN, and ensemble models.

4. **Outputs**
   - Predictions and evaluation results will be saved in their respective folders (e.g., `svm/`, `knn/`, `cnn/`, `ensemble/`).
