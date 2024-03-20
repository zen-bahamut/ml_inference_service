import requests
from sklearn.metrics import classification_report
import datasets
from datetime import datetime, timedelta

API_URL = 'http://api:8080/infer'

def fetch_prediction(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

def main():
    paws_test_data = datasets.load_dataset("paws", "labeled_final")['test']
    predictions = []

    for sample in paws_test_data:
        payload = {'texts': [sample['sentence1'], sample['sentence2']]}
        resp = fetch_prediction(payload)
        if 'predictions' in resp:
            predictions.append(resp['predictions'][0])
        else:
            print(f"Error processing sample: {sample}")
            continue

    true_labels = [int(sample['label']) for sample in paws_test_data]
    report = classification_report(true_labels, predictions)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    print(end - start)
