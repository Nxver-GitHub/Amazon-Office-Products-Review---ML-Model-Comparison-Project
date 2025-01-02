# Surya Pugazhenthi
# Discrete Math 160 - Final Project
# Random Forest Model for Amazon Office Product Review Sentiment Analysis

# Libraries imported
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random
import time
import nltk

# Download the punkt tokenizer
nltk.download('punkt')

# Load a random subset of the data from the JSON file
subset_size = 10000  # Adjust the subset size as needed
with open('Office_Products_Reviews_Data.json', 'r', encoding='utf-8') as file:
    all_reviews = [json.loads(line) for line in file]

# Randomly select a subset of reviews
selected_reviews = random.sample(all_reviews, subset_size)

# Create a DataFrame from the selected subset
df = pd.DataFrame(selected_reviews)

# Randomly select one review for analysis from the subset
random_index = random.randint(0, len(df) - 1)
random_review = df.loc[random_index, 'reviewText']
product_id = df.loc[random_index, 'asin']

# Extract relevant information
reviews = df['reviewText'].astype(str)
sentiments = df['overall']

# Define thresholds for sentiment labels
negative_threshold = 3.5
positive_threshold = 4.5

# Convert numerical sentiment scores to labels
labels = [
    'negative' if s < negative_threshold else
    'neutral' if negative_threshold <= s <= positive_threshold else
    'positive' for s in sentiments
]

# Print message indicating the start of TF-IDF vectorization
print("\nStart TF-IDF Vectorization...")
# Convert text data to numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
print("TF-IDF Vectorization completed.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Print message indicating the start of model training
print("Start Model Training...")
# Train the Random Forest model for classification
model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("Model Training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1)) # Avoid division by zero if no predicted samples for particular class

# Transform the random review using the same TF-IDF Vectorizer
random_review_transformed = vectorizer.transform([random_review])

# Record start time for prediction
start_time = time.time()

# Make a prediction for the randomly selected review
random_prediction = model.predict(random_review_transformed)

# Record end time for prediction
end_time = time.time()

# Print the formatted output with product ID, review, reviewer's rating, and predicted sentiment label
print("\nResult:")
print(f"Product ID: {product_id}")
print("Review:")
# Tokenize the sentences using nltk and join with line breaks
sentences = nltk.sent_tokenize(random_review)
formatted_review = '\n'.join(sentences)
print(formatted_review)

# Print the predicted sentiment label (positive, negative, or neutral)
print("Predicted Sentiment Label:", random_prediction[0])

# Print execution time for prediction
print("Prediction Time:", end_time - start_time, "seconds")

# Wait for user input before closing
input("\nPress Enter to close the program...")