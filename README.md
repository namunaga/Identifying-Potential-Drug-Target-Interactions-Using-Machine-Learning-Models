# Identifying-Potential-Drug-Target-Interactions-Using-Machine-Learning-Models
# 1.Introduction
Drug-target interactions (DTIs) play a pivotal role in drug development and pharmacological research. A DTI refers to the binding of a drug molecule to a specific biological macromolecule (typically a protein), which results in a biological effect. Accurately identifying such interactions is crucial for discovering new drugs, repurposing existing drugs, and understanding disease mechanisms.
Conventionally, identifying DTIs relies heavily on in vitro assays, biochemical screenings, and high-throughput methods. These techniques, while accurate, are costly and time-consuming. The explosion of chemical and biological data has led to an increased interest in computational approaches. Among these, Machine Learning (ML) offers scalable, interpretable, and efficient solutions to infer potential DTIs.
In this study, we focus on a supervised learning approach using a Random Forest classifier. Our methodology uses physicochemical descriptors of drugs and proteins, represented by Morgan fingerprints and amino acid compositions, respectively. We make use of data from ChEMBL v35 to train and validate our model, and employ evaluation metrics like accuracy, F1-score, and ROC-AUC to measure performance.
# 2.Dataset Description
We used data from ChEMBL v35, a manually curated chemical database of bioactive molecules
with drug-like properties. The key datasets included:
• ligands_can.txt: Contains canonical SMILES strings for 68 unique small molecule compounds.
• proteins.txt: Includes amino acid sequences (in FASTA format) for 442 protein targets.
• Y (label matrix): A matrix of size 68x442, where each entry indicates the bioactivity
score (IC50, Ki, Kd) of the drug-target pair.
Each SMILES string describes the molecular structure of a drug, while the amino acid sequences represent the target proteins. The Y matrix provides continuous values representing the binding affinity, which we later binarize for classification purposes.
# 3. Data Preprocessing and Feature Engineering
3.1. Drug Feature Extraction
SMILES (Simplified Molecular Input Line Entry System) strings were converted into Morgan fingerprints using RDKit. Morgan fingerprints are circular binary vectors representing the presence of specific substructures in a molecule. We used:
• Radius: 2
• nBits: 256 (length of the fingerprint vector)
This resulted in a 68 x 256 drug feature matrix. 3.2. Protein Feature Extraction
We computed the amino acid composition (AAC) for each protein sequence. AAC captures the normalized frequency of each of the 20 standard amino acids in a protein. Each sequence was converted into a 20-dimensional feature vector, resulting in a 442 x 20 matrix.
3.3. Pairwise Feature Construction
For every possible drug-target combination (68 drugs x 442 proteins = 30,056 pairs), we concatenated the Morgan fingerprint vector of the drug with the AAC vector of the protein, resulting in 276 features per pair.
3.4. Label Transformation
The Y matrix initially consisted of real-valued bioactivity scores. We binarized these labels using the following criteria:
• Class 1 (Interaction): score ≤ 5.0
• Class 0 (Non-interaction): score > 5.0
This classification threshold is biologically meaningful and is commonly used in DTI prediction studies.
3.5. Addressing Class Imbalance
The dataset was highly imbalanced, with the majority of pairs being non-interactions. We employed SMOTE (Synthetic Minority Oversampling Technique) on the training set to synthetically generate samples of the minority class, thereby balancing the class distribution.
     
# 4. Model and Training Pipeline
We implemented a Random Forest Classifier, which is an ensemble-based learning algorithm that builds multiple decision trees and merges their outputs for more accurate and stable predictions. It operates using the bagging technique (Bootstrap Aggregation), where each tree is trained on a random subset of data with replacement, and predictions are aggregated (via majority voting) to produce the final classification.
Model Configuration:
• Number of Estimators: 100 decision trees were used to ensure a balance between model performance and computational efficiency.
• Train-Test Split: The dataset was split in an 85:15 ratio to allow sufficient data for training while preserving a representative test set for evaluation.
• Stratification: Stratified splitting was applied to maintain the original class distribution (interaction vs. non-interaction) in both the training and testing sets.
• Random State: Set to 42 to ensure reproducibility of results across multiple runs. Why Random Forest?
Random Forest was chosen for several key reasons:
• It handles high-dimensional feature spaces well, which is important as our feature vectors combine 276 dimensions per drug-protein pair.
• It is inherently robust to overfitting, especially when compared to individual decision trees.
• It provides feature importance insights, making it interpretable and useful for biological analysis.
• It works well even with imbalanced datasets when combined with resampling techniques like SMOTE (Synthetic Minority Oversampling Technique).
This configuration enabled the model to learn complex patterns between drug chemical structures (SMILES) and protein sequences (amino acid composition) and generalize them effectively during prediction.
# 5. Evaluation and Results
We evaluated our model using standard classification metrics. Below is the summary:
Classification Report:

 ROC-AUC Score: 0.9176786119301428
The model shows strong predictive performance, especially in identifying interactions (class 1). While performance on the minority class (non-interaction) is relatively lower, SMOTE helped reduce the imbalance effect.
# 6. Visualizations
6.1. ROC Curve (Receiver Operating Characteristic)
The ROC curve is a graphical representation that illustrates the diagnostic ability of a binary classifier as its discrimination threshold varies. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR). A model with perfect prediction has a curve that hugs the top left corner, with an Area Under the Curve (AUC) of 1.0. In our case, the AUC was 0.92, indicating excellent classification performance and strong separation between the interaction and non- interaction classes.
  
Graph Description:
ROC curve—the orange line shows the model’s true positive rate against the false positive rate. The curve closely follows the top-left boundary, which indicates excellent classification capability. AUC score is 0.92, suggesting that the model is highly effective in distinguishing between interaction and non-interaction pairs.
6.2. Confusion Matrix
A confusion matrix provides a tabular summary of actual versus predicted classifications. It helps evaluate the accuracy of a classification model.
• True Positives (TP): Correctly predicted interaction pairs (4363)
• False Positives (FP): Non-interactions incorrectly classified as interactions (60)
• True Negatives (TN): Correctly predicted non-interactions (40)
• False Negatives (FN): Missed actual interactions (46)
From the confusion matrix, we see that the model performs exceptionally well at predicting interactions (Class 1), which is critical in DTI studies where missing a potential interaction could have significant downstream consequences.
• ROC Curve: AUC = 0.92, indicating excellent classification performance.
   
• Confusion Matrix:
o TP = 4363, FP = 60
o TN = 40, FN = 46
• These support high sensitivity and specificity of the classifier
# 7. Conclusion
This project demonstrates that machine learning, particularly Random Forest classifiers, can effectively predict drug-target interactions from bioactivity datasets. With minimal feature engineering (Morgan fingerprints + AAC), we achieved high accuracy and strong ROC-AUC scores. The framework developed here can serve as a foundational pipeline for more complex DTI prediction tasks.
# 8. Future Work
• Incorporate other descriptors such as 3D structure and topological indices
• Explore neural networks and attention-based models for sequence embeddings
• Conduct regression-based scoring using continuous labels from Y
• Benchmark with other models (e.g., LightGBM, SVM, XGBoost)
• Validate predictions against independent DTI databases (e.g., BindingDB, DrugBank)
