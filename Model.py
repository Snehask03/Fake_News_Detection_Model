import pandas as pd
import re
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Load the Fake and True news data
fake_df = pd.read_csv('Fake.csv')  # Replace with the correct path to Fake.csv
true_df = pd.read_csv('True.csv')  # Replace with the correct path to True.csv

# Step 2: Preprocess the data
def clean_text(text):
    """Clean the text by removing special characters, numbers, and extra spaces."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()
    return text

# Clean the text in both datasets
fake_df['text'] = fake_df['text'].apply(clean_text)
true_df['text'] = true_df['text'].apply(clean_text)

# Step 3: Assign labels: 0 for fake, 1 for real
fake_df['label'] = 0  # Fake news is labeled as 0
true_df['label'] = 1  # Real news is labeled as 1

# Combine the datasets
df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], axis=0)

# Step 4: Split data into train and test sets
X = df['text']
y = df['label']

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100  # Max sequence length (to reduce computation time)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Step 6: Build the LSTM model with L2 Regularization, Increased Dropout, and Early Stopping
def create_lstm_model(input_dim, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.02)))  # Reduced LSTM units, increased dropout, stronger L2 regularization
    model.add(Dropout(0.3))  # Additional Dropout Layer to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])  # Lower learning rate
    return model

lstm_model = create_lstm_model(input_dim=5000, max_len=max_len)

# Step 7: Train the LSTM model with Early Stopping and store history
history = lstm_model.fit(X_train_pad, y_train, epochs=15, batch_size=64, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Now you can access the history of training metrics
print(history.history.keys())  # This will give you a list of available metrics in the history

# Step 8: Get LSTM embeddings
lstm_embeddings_train = lstm_model.predict(X_train_pad)
lstm_embeddings_test = lstm_model.predict(X_test_pad)

# Step 9: Train Random Forest on LSTM embeddings
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(lstm_embeddings_train, y_train)

# Step 10: Cross-validation on the Random Forest model
cross_val_scores = cross_val_score(rf_model, lstm_embeddings_test, y_test, cv=5)
print(f"Cross-validation Accuracy: {cross_val_scores.mean():.2f} ± {cross_val_scores.std():.2f}")

# Step 11: Make predictions with the Random Forest model
y_pred = rf_model.predict(lstm_embeddings_test)

# Step 12: Evaluate the model
print("Evaluation on test data:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
