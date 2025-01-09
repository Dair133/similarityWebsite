# Metrics Calculator
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional
import numpy as np
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor
from pdfProcessing.semanticSearch import SemanticScholar
from modelFolder.modelRunner import ModelInference
import os
# backend/app.py
from flask import Flask
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from flask import Blueprint
from dotenv import load_dotenv
import os
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any
import torch

class MetricsCalculator:
    def __init__(self):
        self.batch_size = 8
        self.max_workers = 4  # Adjust based on your CPU
        self.thread_local = threading.local()
        
    def _process_single_paper(self, args) -> Dict:
        """Process a single paper - used for parallel processing"""
        seed_paper, comparison_paper, tokenizer, model = args
        return self.calculate_paper_comparison_metrics(seed_paper, comparison_paper, tokenizer, model)
        
    def process_papers_parallel(self, seed_paper: Dict, papers: List[Dict], tokenizer, model) -> List[Dict]:
        """Process multiple papers in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create args for each paper
            args_list = [(seed_paper, paper, tokenizer, model) for paper in papers]
            # Process papers in parallel
            results = list(executor.map(self._process_single_paper, args_list))
            return results
            
    def batch_get_embeddings(self, abstracts: List[str], tokenizer, model) -> List[torch.Tensor]:
        """Process multiple abstracts in batches"""
        embeddings = []
        for i in range(0, len(abstracts), self.batch_size):
            batch = abstracts[i:i + self.batch_size]
            valid_abstracts = [text for text in batch if text and isinstance(text, str) and text.strip()]
            
            if not valid_abstracts:
                embeddings.extend([torch.zeros(768) for _ in batch])
                continue
                
            with torch.no_grad():
                inputs = tokenizer(
                    valid_abstracts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
            # Add embeddings and zeros for invalid abstracts
            curr_valid_idx = 0
            for text in batch:
                if text and isinstance(text, str) and text.strip():
                    embeddings.append(batch_embeddings[curr_valid_idx])
                    curr_valid_idx += 1
                else:
                    embeddings.append(torch.zeros(768))
                    
        return embeddings

    @torch.no_grad()
    def get_scibert_embedding(self, text: str, tokenizer, model) -> torch.Tensor:
        """Single text processing - optimized version"""
        if not text or not isinstance(text, str) or not text.strip():
            return torch.zeros(768)
            
        try:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            return outputs.last_hidden_state[0, 0, :]
            
        except Exception as e:
            print(f"Error generating SciBERT embedding: {str(e)}")
            return torch.zeros(768)

    def calculate_paper_comparison_metrics(self, seed_paper: Dict[str, Any], comparison_paper: Dict[str, Any], tokenizer, model) -> Dict[str, Any]:
        """Calculate metrics with optimizations"""
        try:
            metrics = {
                'shared_author_count': 0,
                'shared_reference_count': 0,
                'shared_citation_count': 0,
                'reference_cosine': 0.0,
                'citation_cosine': 0.0,
                'abstract_cosine': 0.0
            }
            
            paper_info = comparison_paper['paper_info']
            seed_info = seed_paper['paper_info']
            
            # Process abstract and get embedding
            comp_abstract = paper_info.get('abstract', '')
            comp_embedding = self.get_scibert_embedding(comp_abstract, tokenizer, model)
            paper_info['scibert'] = comp_embedding.tolist()
            
            # Use set operations for faster computation
            seed_authors = frozenset(seed_info.get('authors', []))
            comp_authors = frozenset(paper_info.get('authors', []))
            seed_refs = frozenset(seed_info.get('references', []))
            comp_refs = frozenset(paper_info.get('references', []))
            seed_cites = frozenset(seed_info.get('citations', []))
            comp_cites = frozenset(paper_info.get('citations', []))
            
            # Compute set metrics
            metrics.update({
                'shared_author_count': len(seed_authors & comp_authors),
                'shared_reference_count': len(seed_refs & comp_refs),
                'shared_citation_count': len(seed_cites & comp_cites)
            })
            
            # Calculate cosine similarities in parallel
            if seed_refs and comp_refs:
                metrics['reference_cosine'] = self._calculate_set_similarity(seed_refs, comp_refs)
            if seed_cites and comp_cites:
                metrics['citation_cosine'] = self._calculate_set_similarity(seed_cites, comp_cites)
            
            return metrics
            
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            raise

    def _calculate_set_similarity(self, set1: frozenset, set2: frozenset) -> float:
        """Optimized set similarity calculation"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        if intersection == 0:
            return 0.0
        return intersection / (len(set1) * len(set2)) ** 0.5
        




    def get_relatively_similar_papers(self,papers):
     # Extract scores for analysis
     scores = [p['similarity_score'] for p in papers]
     
     # If we have less than 2 papers, just return them all
     if len(scores) < 2:
        return papers
        
     # Find natural breaks in the data
     sorted_papers = sorted(papers, key=lambda x: x['similarity_score'], reverse=True)
     sorted_scores = [p['similarity_score'] for p in sorted_papers]
    
     # Calculate gaps between consecutive scores
     gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
    
     if len(gaps) > 0:
        # Find the largest gap in the top 75% of papers
        consider_until = max(int(len(gaps) * 0.75), 1)  # Consider at least first gap
        largest_gap = max(gaps[:consider_until])
        gap_index = gaps.index(largest_gap)
        
        # If the gap is significant (more than 5% of the score range)
        score_range = max(scores) - min(scores)
        if largest_gap > score_range * 0.05:
            return sorted_papers[:gap_index + 1]
        else:
            # If no significant gap, use statistical approach
            mean = np.mean(scores)
            std = np.std(scores)
            return [p for p in papers if p['similarity_score'] > (mean + 0.25 * std)]
    
     return papers