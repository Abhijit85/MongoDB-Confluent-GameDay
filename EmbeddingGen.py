import pymongo
# Amazon Bedrock - boto3
import boto3
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get environment variables
mongo_uri = os.getenv('MONGO_URI')
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')



# Setup bedrock
bedrock_runtime = boto3.client( 
    service_name="bedrock-runtime",
    region_name="us-east-1",
    #Passing credentials during client creation
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key= aws_secret_access_key,
    aws_session_token=aws_session_token
 )


# LLM - Amazon Bedrock LLM using LangChain
#from langchain.llms import Bedrock
from langchain.llms import Bedrock

model_id = "anthropic.claude-v2"
model_kwargs =  { 
    "max_tokens_to_sample": 4096,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman:"],
}

# create LLM
llm = Bedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs
)

# Embedding Model - Amazon Titan Embeddings Model using LangChain
from langchain.embeddings import BedrockEmbeddings
#from langchain_community.embeddings import BedrockEmbeddings

# create embeddings
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1"
)


# setup client
client = pymongo.MongoClient(mongo_uri)


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# define db and collection
db = client['aws-gameday']
collection = db.cart_application


# Store the vector embedding back in the database
for doc in collection.find({'description':{"$exists": True}}):
  doc['usecase_embedding'] = bedrock_embeddings.embed_query(doc['description'])
  collection.replace_one({'_id': doc['_id']}, doc)
 