# Chatbot-using-Llama2
# References:

https://docs.pinecone.io/release-notes/2024
https://github.com/pinecone-io
https://github.com/langchain-ai
https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-agent.ipynb#scrollTo=Pa1whr8V3Wfm


## STEP 01- Create a conda environment after opening the repository

conda create -p venv python=3.10 -y
conda activate venv/

## STEP 02- install the requirements

pip install -r requirements.txt

## Create a .env file in the root directory and add your Pinecone credentials as follows:

PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

## Download the quantize model from the link provided in model folder & keep the model in the model directory:

### Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


### From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

## Run the following command
python store_index.py

## Finally run the following command
python app.py

open up localhost:

Techstack Used:

Python
LangChain
Flask
Meta Llama2
Pinecone
