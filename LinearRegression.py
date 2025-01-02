# Surya Pugazhenthi
# Discrete Math 160 - Final Project
# Linear Regression Model for Amazon Office Product Review Sentiment Analysis

# Libraries imported
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import random
import time
import nltk

# Download the punkt tokenizer for sentence formatting
nltk.download('punkt')

# Load a random subset of the data from the JSON file
subset_size = 10000  # Adjust the subset size as needed to train from
with open('Office_Products_Reviews_Data.json', 'r', encoding='utf-8') as file:
    all_reviews = [json.loads(line) for line in file]

# Randomly select a subset of reviews
selected_reviews = random.sample(all_reviews, subset_size)

# Create a DataFrame from the selected subset
df = pd.DataFrame(selected_reviews)

# Randomly select one product review for analysis from the subset
random_index = random.randint(0, len(df) - 1)
random_review = df.loc[random_index, 'reviewText']
product_id = df.loc[random_index, 'asin']

# Extract relevant information
reviews = df['reviewText'].astype(str)
sentiments = df['overall']

# Print message indicating the start of vectorization (TF-IDF)
print("\nStart TF-IDF Vectorization...")
# Convert text data to numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
print("TF-IDF Vectorization completed.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# Print the start of model training
print("Start Model Training...")
# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Print model training is completed
print("Model Training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert regression predictions to discrete sentiment labels
y_pred_labels = [round(prediction) for prediction in y_pred]

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, zero_division=1))

# Transform the random review using the same TF-IDF Vectorizer
random_review_transformed = vectorizer.transform([random_review])

# Record start time for prediction
start_time = time.time()

# Make a prediction for the randomly selected review
random_prediction = model.predict(random_review_transformed)

# Record end time for prediction
end_time = time.time()

# Print the formatted output with product ID, review, and predicted sentiment score (1 to 10)
print("\nResult:")
print(f"Product ID: {product_id}")
print("Review:")
# Tokenize the sentences using nltk and join with line breaks
sentences = nltk.sent_tokenize(random_review)
formatted_review = '\n'.join(sentences)
print(formatted_review)

# random_prediction[0] contains the predicted sentiment 
original_min = 1  # Minimum value of the 5-star scale
original_max = 5  # Maximum value of the 5-star scale

new_min = 1  # Minimum value of the predicted sentiment score  
new_max = 10  # Maximum value of the predicted sentiment score

# Linear transformation
scaled_prediction = ((random_prediction[0] - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min

# Lock the scaled prediction between 1 and 10
scaled_prediction = max(1, min(scaled_prediction, 10))

# Round the locked scaled prediction to an integer
rounded_prediction = round(scaled_prediction)
# Print the predicted sentiment score
print("Predicted Sentiment Score (1 to 10):", rounded_prediction)

# Print execution time for prediction
print("Prediction Time:", end_time - start_time, "seconds")

# Wait for user input before closing
input("\nPress Enter to close the program...")