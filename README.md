# SDP-BaseOn-Svm
# Software Defect Prediction Model

This project implements a software defect prediction model using Random Forest, Support Vector Machine, and k-Nearest Neighbors algorithms. The goal is to extract code vectors from Java projects and predict defects in project files.

## Requirements

Ensure you have the following dependencies installed:

- `numpy~=1.21.5`
- `yaml~=0.2.5`
- `pyyaml~=6.0`
- `scikit-learn~=1.1.1`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation:**
   - Prepare your Java project files and ensure they are structured properly.

2. **Feature Extraction:**
   - Utilize code vectorization techniques to extract features from the Java project files. This step involves transforming code into numerical representations suitable for machine learning algorithms.

3. **Model Training:**
   - Train the software defect prediction models (Random Forest, Support Vector Machine, and k-Nearest Neighbors) using the extracted features.

4. **Defect Prediction:**
   - Use the trained models to predict defects in Java project files. Input the code vectors into the models and analyze the predictions to identify potential defects.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ikkkp/SDP-BaseOn-Svm
   cd <repository_folder>
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the code for feature extraction and model training.

   main.py

4. output

   result.txt  

5. Utilize the trained models for defect prediction in your Java project files.

    A project to compute all kinds of descriptors for those software products(e.g. LOC, McCabe, Halstead).

    This project leverages Antlr4 and JavaParser for syntax tree level analysis, enabling accurate and comprehensive information extraction from code. Its highly extensible nature allows the calculation of various language code metrics. Currently, it supports the Java programming language.

   https://github.com/jaki2012/SoftwareMetricsAnalyse
    
   ```bash
    LOC_BLANK,
    BRANCH_COUNT,
    CALL_PAIRS,
    LOC_CODE_AND_COMMENT,
    LOC_COMMENTS,
    CONDITION_COUNT,
    CYCLOMATIC_COMPLEXITY,
    CYCLOMATIC_DENSITY,
    DECISION_COUNT,
    DECISION_DENSITY,
    DESIGN_COMPLEXITY,
    DESIGN_DENSITY,
    EDGE_COUNT,
    ESSENTIAL_COMPLEXITY,
    ESSENTIAL_DENSITY,
    LOC_EXECUTABLE,
    PARAMETER_COUNT,
    GLOBAL_DATA_COMPLEXITY,
    GLOBAL_DATA_DENSITY,
    HALSTEAD_CONTENT,
    HALSTEAD_DIFFICULTY,
    HALSTEAD_EFFORT,
    HALSTEAD_ERROR_SET,
    HALSTEAD_LENGTH,
    HALSTEAD_LEVEL,
    HALSTEAD_PROGRAM_TIME,
    HALSTEAD_VOLUME,
    MAINTENANCE_SEVERITY,
    MODIFIED_CONDITION_COUNT,
    MULTIPLE_CONDITION_COUNT,
    NODE_COUNT,
    NORMALIZED_CYCLOMATIC_COMPLEXITY,
    NUM_OPERANDS,
    NUM_OPERATORS,
    NUM_UNIQUE_OPERANDS,
    NUM_UNIQUE_OPERATORS,
    NUMBER_OF_LINES,
    PERCENT_COMMENTS,
    LOC_TOTAL
   ```
    
## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated.

## License

This project is licensed under the [MIT License](LICENSE).