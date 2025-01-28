# Sentiment_Analysis_Pipeline

## Data Acquisition
- Download the dataset from [kaggle](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews?select=IMDB+Dataset.csv)
- Place it in the root folder as `IMDB Dataset.csv`.

## Project Setup
1. **Clone this repository**
```bash
git clone https://github.com/pdobariya1/Sentiment_Analysis_Pipeline.git
```

2. **Change directory**
```bash
cd Sentiment_Analysis_Pipeline
```

3. **Create virtual environment**
```bash
conda create -p {env_name} python=3.10.16 -y
```

4. **Activate Environment**
```bash
conda activate ./{env_name}
```

5. **Install dependencies**
```bash
pip install -r requirements.txt --use-pep517
```

6. **Run the Flask server**
```bash
python app.py
```

## Test the Endpoint
```python
import requests
url = "http://127.0.0.1:5000/predict"
data = {"review_text": "This movie was amazing!"}
response = requests.post(url, json=data)
print(response.json())
```