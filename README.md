# Fake_News_Detection_Model

# Fake News Detection using LSTM & Random Forest  

This project implements a **hybrid approach** for detecting fake news using the **ISOT Fake News Dataset**. It combines an **LSTM model** for text feature extraction with a **Random Forest classifier** for final classification, improving accuracy and robustness.  

## Overview  
Fake news is a growing concern in the digital age, and this project aims to build a **deep learning-based** detection system using **Natural Language Processing (NLP)** techniques. The model is trained on labeled **fake** and **real** news articles, distinguishing between them using advanced machine learning techniques.  

## Features  
- **Preprocessing Pipeline** – Text cleaning, tokenization, and sequence padding.  
- **LSTM Model** – Extracts meaningful text embeddings for classification.  
- **Random Forest Classifier** – Learns patterns from LSTM embeddings for final classification.  
- **Cross-Validation** – Ensures model robustness using k-fold validation.  
- **Early Stopping & Regularization** – Prevents overfitting using dropout, L2 regularization, and reduced learning rates.  

## Dataset  
This project uses the **ISOT Fake News Dataset**, which contains two categories of news articles:  
- **Fake News** (`Fake.csv`) – Misleading or false articles.  
- **True News** (`True.csv`) – Legitimate, verified news articles.  

Each article contains a **title** and **text**, with labels assigned:  
- `0` → Fake News  
- `1` → Real News  
