# backend/routes/upload_routes.py
# APP.py script
import logging
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
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
    
import urllib3
import json
import time
import numpy as np

import socket
import json
import time
import numpy as np
import ssl
class APIPersonalPCClass:
    def __init__(self):
        pass
    
    def send_batch_scibert_request(self, abstracts_dict, base_url="https://fypserver.ngrok.app"):
     # Extract hostname from URL
     hostname = base_url.replace("https://", "").replace("http://", "").split("/")[0]
     port = 443  # HTTPS port
    
     print(f"Sending batch request to {hostname}:{port} using raw sockets...")
     start_time = time.time()
     
     try:
        # Create raw socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(15)  # 15 second timeout for batch processing
        
        # Wrap with SSL for HTTPS
        context = ssl.create_default_context()
        wrapped_socket = context.wrap_socket(sock, server_hostname=hostname)
        
        # Connect to the server
        wrapped_socket.connect((hostname, port))
        print(f"Socket connected in {time.time() - start_time:.2f} seconds")
        
        # Convert dict to JSON and prepare the HTTP POST request
        payload = json.dumps({'texts':abstracts_dict})
        request = (
            f"POST /batch_embed HTTP/1.1\r\n"
            f"Host: {hostname}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{payload}"
        )
        
        # Send the request
        send_start = time.time()
        wrapped_socket.sendall(request.encode('utf-8'))
        print(f"Request sent in {time.time() - send_start:.2f} seconds")
        
        # Receive the response
        recv_start = time.time()
        response_data = b""
        while True:
            chunk = wrapped_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        print(f"Response received in {time.time() - recv_start:.2f} seconds")
        
        # Close the socket
        wrapped_socket.close()
        
        # Parse the response
        response_text = response_data.decode('utf-8')
        
        # Check if response contains success status code
        if "200 OK" in response_text:
            # Extract JSON response from the HTTP response
            try:
                json_start = response_text.find('{')
                if json_start >= 0:
                    json_data = json.loads(response_text[json_start:])
                    print(f"Successfully processed batch request")
                    print(f"Total time: {time.time() - start_time:.2f} seconds")
                    return json_data
                else:
                    print("Could not find JSON in response")
                    return None
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                return None
        else:
            print(f"Server returned non-200 status. Response: {response_text[:100]}")
            return None
            
     except socket.timeout:
        print(f"Socket operation timed out after {time.time() - start_time:.2f} seconds")
        return None
     except socket.error as e:
        print(f"Socket error: {e}")
        return None
     except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return None


    def test_local_server(self, abstract_text, base_url="https://fypserver.ngrok.app"):
     # Extract hostname from URL
     hostname = base_url.replace("https://", "").replace("http://", "").split("/")[0]
     port = 443  # HTTPS port
    
     print(f"Connecting directly to {hostname}:{port}...")
     start_time = time.time()
    
     try:
        # Create raw socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout
        
        # Wrap with SSL for HTTPS
        context = ssl.create_default_context()
        wrapped_socket = context.wrap_socket(sock, server_hostname=hostname)
        
        # Connect to the server
        wrapped_socket.connect((hostname, port))
        print(f"Socket connected in {time.time() - start_time:.2f} seconds")
        
        # Prepare the HTTP request with minimal headers
        payload = json.dumps({"text": abstract_text})
        request = (
            f"POST /embed HTTP/1.1\r\n"
            f"Host: {hostname}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{payload}"
        )
        
        # Send the request
        send_start = time.time()
        wrapped_socket.sendall(request.encode('utf-8'))
        print(f"Request sent in {time.time() - send_start:.2f} seconds")
        
        # Receive the response
        recv_start = time.time()
        response_data = b""
        while True:
            chunk = wrapped_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        print(f"Response received in {time.time() - recv_start:.2f} seconds")
        
        # Close the socket
        wrapped_socket.close()
        
        # Parse the HTTP response
        response_text = response_data.decode('utf-8')
        headers, body = response_text.split('\r\n\r\n', 1)
        
        # Extract JSON from body (ignoring potential transfer-encoding chunking for simplicity)
        try:
            # Try to find and parse the JSON part of the response
            json_start = body.find('{')
            if json_start >= 0:
                json_data = json.loads(body[json_start:])
                if "embedding" in json_data:
                    embedding = json_data["embedding"]
                    print(f"Successfully extracted embedding, dimension: {len(embedding)}")
                    return embedding
            
            print("Could not find embedding in response")
            return [0.0] * 768  # Default size
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            return [0.0] * 768
            
     except socket.timeout:
        print(f"Socket operation timed out after {time.time() - start_time:.2f} seconds")
        return [0.0] * 768
     except Exception as e:
        print(f"Exception: {type(e).__name__}: {str(e)}")
        return [0.0] * 768
    
    
    
    def check_server_health(self, base_url="https://fypserver.ngrok.app"):
     # Extract hostname from URL
     hostname = base_url.replace("https://", "").replace("http://", "").split("/")[0]
     port = 443  # HTTPS port
    
     print(f"Checking server health at {hostname}:{port} using raw sockets...")
     start_time = time.time()
    
     try:
        # Create raw socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        
        # Wrap with SSL for HTTPS
        context = ssl.create_default_context()
        wrapped_socket = context.wrap_socket(sock, server_hostname=hostname)
        
        # Connect to the server
        wrapped_socket.connect((hostname, port))
        print(f"Socket connected in {time.time() - start_time:.2f} seconds")
        
        # Prepare a simple HTTP GET request
        request = (
            f"GET /health HTTP/1.1\r\n"
            f"Host: {hostname}\r\n"
            f"Connection: close\r\n\r\n"
        )
        
        # Send the request
        send_start = time.time()
        wrapped_socket.sendall(request.encode())
        print(f"Request sent in {time.time() - send_start:.2f} seconds")
        
        # Receive the response
        recv_start = time.time()
        response_data = b""
        while True:
            chunk = wrapped_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        print(f"Response received in {time.time() - recv_start:.2f} seconds")
        
        # Close the socket
        wrapped_socket.close()
        
        # Parse the response to check if it's healthy
        response_text = response_data.decode('utf-8')
        
        # Check if response contains success status code
        if "200 OK" in response_text:
            print(f"Server is healthy! Response: {response_text}")
            print(f"Total time: {time.time() - start_time:.2f} seconds")
            return True
        else:
            print(f"Server returned non-200 status. Response: {response_text[:100]}")
            return False
            
     except socket.timeout:
        print(f"Socket operation timed out after {time.time() - start_time:.2f} seconds")
        return False
     except socket.error as e:
        print(f"Socket error: {e}")
        return False
     except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return False