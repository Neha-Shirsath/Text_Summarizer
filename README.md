# AI Text Summarizer

An AI-powered text summarization web application built using FastAPI,
Hugging Face Transformers, PyTorch, HTML, CSS, and JavaScript.

The application uses a fine-tuned T5 model for generating concise
summaries from input text.

## Features

- AI-based text summarization
- Fine-tuned T5 Transformer model
- FastAPI backend
- Simple web interface
- REST API endpoint
- Hugging Face model hosting

## Tech Stack

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- T5
- HTML
- CSS
- JavaScript

## Model

The trained model is hosted on Hugging Face:

https://huggingface.co/neha3008/text-summarizer-t5

The application automatically downloads the model when it starts.

## Project Structure

Text_Summarizer/
│
├── app.py

├── requirements.txt

├── .gitignore

├── static/
│   └── style.css

└── templates/
    └── index.html

## Installation

Clone the repository: git clone https://github.com/Neha-Shirsath/Text_Summarizer.git

cd Text_Summarizer

Create a virtual environment: python -m venv myenv

Activate it on Windows: myenv\Scripts\activate

Install dependencies: pip install -r requirements.txt

## Run the Application

uvicorn app:app --reload

Open:

http://127.0.0.1:8000

## API

POST /summarize/

Example request:

{
    "dialogue": "Enter your text here..."
}

Example response:

{
    "summary": "Generated summary..."
}

## Deployment

The application is configured for deployment using FastAPI.

The trained model is hosted separately on Hugging Face because the
model file is too large to store directly in GitHub.

## Author

Neha Shirsath
