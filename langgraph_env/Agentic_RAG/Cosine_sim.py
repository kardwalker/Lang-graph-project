from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
import numpy as np

embed_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/", 
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key
    deployment="text-embedding-ada-002",  # Deployment name must match what you
    # created in Azure
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error)
)

text1 = "LangGraph is a library for building stateful, multi-actor applications with LLMs."

text2 = "LangChain is a framework for building context-aware reasoning applications."

text3 = "langgraph is a framework for building stateful agent with persistence,  HITL and multi-system agentic applications." \
    

emb1 , emb2 , emb3 = embed_model.embed_documents([text1, text2, text3])
print("Embedding 1:", emb1[:10])
print("embedding 2 " , emb2[:10])
print("embedding 3", emb3[:10])
print(type(emb1))

async def cosine_simi(vec1 , vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    response = np.dot(vec1 , vec2)/(norm1*norm2)

    return response


async def main():
    simi_1_and_2 = await cosine_simi(emb1 , emb2)
    simi_2_and_3 = await cosine_simi(emb2 , emb3)
    simi_3_and_1 = await cosine_simi(emb3 , emb1)

    print("Cosine similarity between text1 and text2" ,simi_1_and_2)
    print("Cosine similarity between text 2 and text 3" , simi_2_and_3)
    print("Cosine similarity between text 3 and text 1" , simi_3_and_1)

# Run the async function
import asyncio
asyncio.run(main())
"""
async def main():
    results = await asyncio.gather(
        cosine_simi(emb1, emb2),
        cosine_simi(emb2, emb3),
        cosine_simi(emb3, emb1)
    )
    
    simi_1_and_2, simi_2_and_3, simi_3_and_1 = results
    # ... print statements
"""

# Cosine distance

# Calculated as (1 - similarity)
# Same direction, 0.0
# If prependicular, 1.0
# If completely opposite, 2.0