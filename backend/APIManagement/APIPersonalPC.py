# backend/routes/upload_routes.py
# APP.py script
import logging
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any
import numpy as np
import requests
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor

# Import for model runners
from modelFolder.modelRunners.standardModelRunner32k3 import ModelInference
from modelFolder.metricsCalculator import MetricsCalculator
from pdfProcessing.SearchTermCache import SearchTermCache
from pdfProcessing.localDatabaseManager import LocalDatabaseManager
import os
# backend/app.py
from flask import Flask
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from flask import Blueprint
from dotenv import load_dotenv
import os
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor

class APIPersonalPCClass:
    def __init__(self):
        pass
    
    
    def test_local_server(self, abstract_text, url="http://localhost:5000/embed"):
     """
    Test the SciBERT embedding server by sending an abstract and printing results
    
    Args:
        abstract_text (str): The abstract text to embed
        url (str): The server URL endpoint
    """
    # Prepare the request payload
     payload = {
        "text": abstract_text
    }
    
     # Send the request to the server
     print(f"Sending request to {url}...")
     start_time = time.time()
    
     try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Calculate request time
        elapsed_time = time.time() - start_time
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            embedding = np.array(result["embedding"])
            
            print(f"Request successful! Status code: {response.status_code}")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            print(f"Embedding dimension: {result['dimension']}")
            print(f"Embedding type: {type(embedding)}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
            print(f"Last 5 values: {embedding[-5:]}")
            print(f"Min value: {np.min(embedding)}")
            print(f"Max value: {np.max(embedding)}")
            print(f"Mean value: {np.mean(embedding)}")
            
            return embedding
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
     except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None