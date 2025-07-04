import joblib
from flask import Flask, request, render_template

import re # For regular expressions in preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK resources (run once if not already downloaded in this env) ---
# It's best to run these downloads in a separate script or directly in your
# environment's Python interpreter, not typically in your app.py, to avoid
# repeated downloads or potential blocking issues during app startup.
# nltk.download('stopwords')
# nltk.download('wordnet')
# -------------------------------------------------------------------------------

# --- Load the Model and Vectorizer ---
# Make sure these paths are correct relative to where app.py is run
try:
    model = joblib.load('final_spam_detector_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and Vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: Model or Vectorizer file not found. Make sure 'final_spam_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    exit() # Exit if models can't be loaded

# Initialize NLTK components for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Define the preprocessing function (MUST BE IDENTICAL TO TRAINING) ---
def preprocess_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split() # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Lemmatize and remove stop words
    return ' '.join(tokens)


# Initialize the Flask application
app = Flask(__name__)

# --- Define the home page route ---
@app.route('/')
def home():
    # This function will render our HTML template for the home page
    return render_template('index.html') # We'll create index.html next

# --- Define the prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the email text from the form
        email_text = request.form['email_content']

        # --- IMPORTANT: Apply the SAME preprocessing as during training ---
        processed_email_text = preprocess_text(email_text)

        # Transform the input text using the loaded TF-IDF vectorizer
        # Make sure to call .transform, not .fit_transform, as the vectorizer is already fitted
        text_vectorized = tfidf_vectorizer.transform([processed_email_text])

        # Make a prediction using the loaded model
        prediction = model.predict(text_vectorized)[0] # [0] to get the single prediction value

        # Interpret the prediction
        result = "SPAM" if prediction == 1 else "NOT SPAM"

        # Pass the result back to the HTML template
        return render_template('index.html', prediction_text=f'The email is: {result}')

# --- Run the Flask application ---
if __name__ == '__main__':
    # debug=True allows for automatic reloading when code changes
    # and provides detailed error messages (useful for development)
    app.run(debug=True)