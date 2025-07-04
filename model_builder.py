import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB # Added for comparison
from sklearn.svm import LinearSVC # Added for comparison
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK resources (run once) ---
# nltk.download('stopwords')
# nltk.download('wordnet')
# ----------------------------------------

# Load the dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: spam.csv not found. Make sure it's in the correct directory.")
    exit()

# Initial data cleaning and renaming
df = df[['v1', 'v2']]
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

print("\n--- Starting Enhanced Text Preprocessing ---")

# Initialize WordNetLemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split() # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Lemmatize and remove stop words
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete!")

# --- Step 4: Feature Extraction (Text Vectorization) ---
print("\n--- Step 4: Feature Extraction (Text Vectorization) ---")

# Initialize TF-IDF Vectorizer - use the processed text
# Experiment with ngram_range for capturing phrases
tfidf_vectorizer = TfidfVectorizer(min_df=1, lowercase=False, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['target']

print(f"Shape of features (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")
print(f"Number of unique words (features) extracted: {X.shape[1]}")

# --- Step 5: Building and Training the Model ---
print("\n--- Step 5: Building and Training the Model ---")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Added stratify=y to ensure similar proportion of ham/spam in train and test sets

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- Option A: Logistic Regression with GridSearchCV ---
print("\n--- Training Logistic Regression with GridSearchCV ---")
param_grid_lr = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear'], # 'lbfgs' can also be tried, but needs more max_iter
    'penalty': ['l1', 'l2']
}
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
                             param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_lr.fit(X_train, y_train)

print(f"\nLogistic Regression Best parameters: {grid_search_lr.best_params_}")
print(f"Logistic Regression Best cross-validation accuracy: {grid_search_lr.best_score_:.4f}")
best_lr_model = grid_search_lr.best_estimator_

# --- Option B: Multinomial Naive Bayes ---
print("\n--- Training Multinomial Naive Bayes Model ---")
mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)

# --- Option C: LinearSVC (SVM) ---
print("\n--- Training LinearSVC (Support Vector Machine) Model ---")
svc_model = LinearSVC(random_state=42, C=0.5, max_iter=2000) # C can be tuned with GridSearchCV too
svc_model.fit(X_train, y_train)

print("\nModel training complete for all options!")

# --- Step 6: Evaluating the Models ---
print("\n--- Step 6: Evaluating the Models ---")

models = {
    "Logistic Regression (Tuned)": best_lr_model,
    "Multinomial Naive Bayes": mnb_model,
    "LinearSVC": svc_model
}

for name, model in models.items():
    print(f"\n--- Evaluation for {name} ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

    print(f"Accuracy Score: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

# Determine the best model based on accuracy (or F1-score for imbalanced data)
# You might need to manually compare the reports, or you could automate this.
# For simplicity, let's just pick one to save based on expected good performance.
# Often LinearSVC or tuned Logistic Regression perform very well on this dataset.
final_model_to_save = best_lr_model # You can change this to mnb_model or svc_model based on results

print("\nModel evaluation complete. Next, we save the best model and vectorizer!")

# --- Step 7: Saving the Best Model and Vectorizer ---
print("\n--- Step 7: Saving the Model and Vectorizer ---")

model_path = 'final_spam_detector_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

joblib.dump(final_model_to_save, model_path)
print(f"Best Model saved to: {model_path}")

joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"Vectorizer saved to: {vectorizer_path}")

print("\nModel and Vectorizer saved successfully. Ready for Web Integration!")