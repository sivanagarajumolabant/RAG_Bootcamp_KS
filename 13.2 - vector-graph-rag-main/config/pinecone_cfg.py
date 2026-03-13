from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()


def get_pinecone() -> Pinecone:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pc
