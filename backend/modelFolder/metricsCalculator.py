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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any
import torch

class MetricsCalculator:
    def __init__(self):
        self.batch_size = 8
        self.max_workers = 4  
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
            
            # Convert lists of strings to one long string for references and citations
            seed_references = ' '.join(seed_info.get('references', [])) if isinstance(seed_info.get('references', []), list) else seed_info.get('references', '')
            comp_references = ' '.join(paper_info.get('references', [])) if isinstance(paper_info.get('references', []), list) else paper_info.get('references', '')
            seed_citations = ' '.join(seed_info.get('citations', [])) if isinstance(seed_info.get('citations', []), list) else seed_info.get('citations', '')
            comp_citations = ' '.join(paper_info.get('citations', [])) if isinstance(paper_info.get('citations', []), list) else paper_info.get('citations', '')
            
            # Calculate cosine similarities in parallel
            if seed_references and comp_references:
                # print('calculating cosines for refs', seed_references, comp_references)
                metrics['reference_cosine'] = self.calculate_text_cosine(seed_references, comp_references)
            if seed_citations and comp_citations:
                # print('calculating cosines for cites', seed_citations, comp_citations)
                metrics['citation_cosine'] = self.calculate_text_cosine(seed_citations, comp_citations)
            
            # Calculate abstract cosine similarity
            seed_abstract = seed_info.get('abstract', '')
            comp_abstract = paper_info.get('abstract', '')
            if seed_abstract and comp_abstract:
                # print('calculating cosines for abstracts', seed_abstract, comp_abstract)
                metrics['abstract_cosine'] = self.calculate_text_cosine(seed_abstract, comp_abstract)
                
            # print('cosines are ', metrics['reference_cosine'], metrics['citation_cosine'])
            
            return metrics
            
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            raise


        
    def calculate_text_cosine(self, text1, text2):
     """Calculate cosine similarity between two text strings using TF-IDF."""
     if not isinstance(text1, str) or not isinstance(text2, str):
        print('text1 and text2 are not strings, they are ', type(text1), type(text2))
        return 0
    
    # Clean and validate texts
     text1 = text1.strip().lower()
     text2 = text2.strip().lower()
     if not text1 or not text2:
        print('text1 or text2 is empty')
        return 0
    
    # Use TF-IDF for cosine similarity
     try:
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
     except Exception as e:
        print(f"Error in text cosine calculation: {str(e)}")
        return 0



    def get_relatively_similar_papers(self, papers, min_results=5, max_results=20):
    # First, ensure we have valid similarity scores and can access them
     try:
        # Handle both dictionary and list formats
        if isinstance(papers, list):
            # Make sure each paper has a similarity_score
            papers = [p for p in papers if isinstance(p, dict) and 'similarity_score' in p]
        else:
            print("Papers input is not a list, type:", type(papers))
            return []
            
        # If we have very few papers, return them all
        if len(papers) <= min_results:
            return papers
        
        # Sort papers by similarity score
        sorted_papers = sorted(papers, key=lambda x: float(x.get('similarity_score', 0)), reverse=True)
        sorted_scores = [float(p.get('similarity_score', 0)) for p in sorted_papers]
        
        # Calculate gaps between consecutive scores
        gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
        
        if len(gaps) > 0:
            # Consider a larger portion of papers for gap analysis (90%)
            consider_until = max(int(len(gaps) * 0.9), min_results)
            largest_gap = max(gaps[:consider_until])
            gap_index = gaps.index(largest_gap)
            
            # Make gap significance threshold more lenient (3%)
            score_range = max(sorted_scores) - min(sorted_scores)
            if largest_gap > score_range * 0.03:
                # Ensure we return at least min_results papers
                result_count = max(gap_index + 1, min_results)
                result_count = min(result_count, max_results)  # But not more than max_results
                return sorted_papers[:result_count]
            else:
                # More lenient statistical approach
                mean = np.mean(sorted_scores)
                std = np.std(sorted_scores)
                
                # Try increasingly lenient thresholds until we get enough results
                for threshold_multiplier in [0.0, -0.25, -0.5, -0.75, -1.0]:
                    threshold = mean + threshold_multiplier * std
                    filtered_papers = [p for p in sorted_papers if float(p.get('similarity_score', 0)) > threshold]
                    
                    if len(filtered_papers) >= min_results:
                        # If we have enough papers, return them (up to max_results)
                        return filtered_papers[:max_results]
                
                # If we still don't have enough papers, just return top min_results
                return sorted_papers[:min_results]
        
        # If no gaps (unlikely), return top papers up to max_results
        return sorted_papers[:max_results]
        
     except Exception as e:
        print("Error in get_relatively_similar_papers:", str(e))
        # Add some debug printing
        print("Papers type:", type(papers))
        if isinstance(papers, list) and len(papers) > 0:
            print("First paper type:", type(papers[0]))
            print("First paper content:", papers[0])
        return []
 
 
 
 
    def apply_source_weights(self, papers: List[Dict]) -> Dict:
    # Define weights for different search types
     weights = {
        'core_concept': 1.00,         
        'core_methodology': 1.00,   
        'related_methodology': 1.00,  
        'abstract_concept': 0.1,      
        'cross_domain_applications': 0.1,  
        'theoretical_foundation': 0.1, 
        'analogous_problems': 0.1    
    }
    
     weighted_papers = []
     for paper in papers:
        weighted_paper = paper.copy()
        sources = paper.get('source_info', [])
        
        if sources and len(sources) == 1:  # Since each paper has exactly one search type
            source = sources[0]  # Get the single source
            weight = weights.get(source['search_type'], 1.0)
            
            # print('max weight is ', weight,'for search type', paper.get('source_info', []))
            original_score = paper.get('similarity_score', 0)
            # print('before weight is ', original_score)
            
            weighted_paper['similarity_score'] = original_score * weight
            # print('after weight is ', weighted_paper['similarity_score'])
            
            weighted_paper['applied_weight'] = weight
            
        weighted_papers.append(weighted_paper)
    
    # Sort papers by adjusted similarity score
     weighted_papers.sort(
        key=lambda x: x.get('similarity_score', 0),
        reverse=True
    )
    
    # Return dictionary with 'compared_papers' key
     return {'compared_papers': weighted_papers}