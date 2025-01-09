import os
import torch
from flask import Flask, request, jsonify
from transformers import RagTokenizer
from models_utils import get_rag_model, get_device
from data_utils import create_data_loader

app = Flask(__name__)

# Load the model and tokenizer
model, tokenizer = get_rag_model()
device = get_device()
model.to(device)

# Load the dataset and create a data loader
data_file = "hindu_scriptures.csv"
batch_size = 32
data_loader = create_data_loader(data_file, tokenizer, batch_size)

@app.route("/generate", methods=["POST"])
def generate_scripture():
    input_text = request.json["input_text"]
    max_length = request.json.get("max_length", 512)

    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    # Move the inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the scripture
    output = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length)

    # Decode the output
    generated_scripture = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_scripture": generated_scripture})

@app.route("/train", methods=["POST"])
def train_model():
    # Train the model on the dataset
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

    model.eval()

    return jsonify({"message": "Model trained successfully"})

if __name__ == "__main__":
    app.run(debug=True)