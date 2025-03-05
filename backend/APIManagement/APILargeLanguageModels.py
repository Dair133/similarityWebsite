# backend/routes/upload_routes.py
# APP.py script
import logging
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any
import numpy as np
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor

# Import for model runners
from modelFolder.modelRunners.standardModelRunner32k3 import ModelInference
from modelFolder.metricsCalculator import MetricsCalculator
from pdfProcessing.SearchTermCache import SearchTermCache
from pdfProcessing.localDatabaseManager import LocalDatabaseManager
from PromptManager import PromptManager
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
import anthropic


class APILargeLanguageModelsClass:
    def __init__(self):
        pass
    def ask_claude(self,pdfText: str, systemInstructions: str, api_key: str) -> str:
        try:
            message = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "system": systemInstructions,
            "messages": [
                {
                    "role": "user",
                    "content": f"{pdfText}"
                }
            ]
        }
        
            client = anthropic.Anthropic(api_key=api_key)
        
        # Get the response
            response = client.messages.create(**message)
        
            return response
        
        except Exception as e:
            # If something goes wrong, log it and raise the error
            self.logger.error(f"Error communicating with Claude: {str(e)}")
            raise