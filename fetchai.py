# First, ensure you have the necessary libraries installed:
pip install fetchai==0.11.0 transformers==4.14.2

import fetchai.ledger.api as ledger_api
import requests
from transformers import pipeline

# Initialize Fetch.ai ledger API
ledger = ledger_api.FactomLedgerApi()

# Hugging Face pipeline for text classification
nlp_pipeline = pipeline("zero-shot-classification")

def fetchai_fetch_data(topic):
    """
    Fetch data related to a topic from Fetch.ai decentralized storage.
    """
    data = ledger.get_data(topic)
    return data

def hugging_face_classify(text, labels):
    """
    Use Hugging Face pipeline for zero-shot classification.
    """
    result = nlp_pipeline(text, labels)
    return result

def main():
    # Fetch topic from user
    topic = input("Enter the topic you want to learn about: ")

    # Fetch data from Fetch.ai decentralized storage
    fetched_data = fetchai_fetch_data(topic)

    if fetched_data:
        print("Data retrieved successfully from Fetch.ai:")
        print(fetched_data)

        # Prompt user to ask a question
        question = input("Ask a question about the retrieved data: ")

        # Perform zero-shot classification using Hugging Face pipeline
        labels = ["yes", "no"]  # Example labels for classification
        classification_result = hugging_face_classify(question, labels)

        # Print classification result
        print("Classification Result:", classification_result)
    else:
        print("No data found on the provided topic.")

if __name__ == "__main__":
    main()
