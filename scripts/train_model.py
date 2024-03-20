from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import pandas as pd


dataset = load_dataset("paws", "labeled_final")

train_data = dataset["train"]
test_data = dataset["validation"]


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


training_args = TrainingArguments(
    output_dir='./results',  
    num_train_epochs=1,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=32,  
    warmup_steps=500, 
    weight_decay=0.01,  
    logging_dir='./logs', 
)


def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length')


train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)


metric = load_metric("accuracy")

trainer = Trainer(
    model=model,  
    args=training_args,  
    train_dataset=train_data,  
    eval_dataset=test_data, 
    compute_metrics=lambda p: metric.compute(predictions=p.predictions.argmax(axis=1), references=p.label_ids),
)


trainer.train()


model_artifact_dir = "model_artifact"
trainer.save_model(model_artifact_dir)
tokenizer.save_pretrained(model_artifact_dir)

print("Model artifact saved.")
