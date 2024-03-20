from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

model_path = "john-adrian/bert-paw-public"
tokenizer = AutoTokenizer.from_pretrained("john-adrian/bert-paws-public")
model = AutoModelForSequenceClassification.from_pretrained("john-adrian/bert-paws-public")
model.eval()

async def perform_inference(texts):
    inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
    return predictions

@app.route('/infer', methods=['POST'])
async def infer():
    data = request.get_json()
    texts = data['texts']
    predictions = await perform_inference(texts)
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
