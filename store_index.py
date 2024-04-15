from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY") 
pc = pinecone.Pinecone(api_key=api_key) 


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

index_name ='medicalchatbot'


if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

index_name="medicalchatbot"

if index_name not in pc.list_indexes().names():
    # create the index if it does not exist

    pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# wait for index to be initialized
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)  
index.describe_index_stats()

#Creating Embeddings for Each of The Text Chunks & storing
for i, t in zip(range(len(text_chunks)), text_chunks):
   query_result = embeddings.embed_query(t.page_content)

   index.upsert(
   vectors=[
        {
            "id": str(i),  # Convert i to a string
            "values": query_result, 
            "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
        }
    ]
)
   
index.describe_index_stats() 

#pc.delete_index(index_name)