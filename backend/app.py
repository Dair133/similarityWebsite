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
from pdfProcessing.semanticSearch import SemanticScholar

# Import for model runners
from modelFolder.modelRunners.standardModelRunner37k2 import ModelInference
from modelFolder.modelRunners.lowSciBertModelRunner import LowSciBertModelInference  # Import the new model inference class
from modelFolder.modelRunners.standardModelRunner37k3 import ModelInferenceThree

from modelFolder.metricsCalculator import MetricsCalculator
from pdfProcessing.SearchTermCache import SearchTermCache

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

claudeInstruction_extractTitleMethodInfo = """
Extract the paper title and core methodologies. Each methodology should combine a fundamental technique with 1-2 scope-defining terms to ensure relevant but not overly narrow results.
IMPORTANT:The resulting search result should provide upon using it for a text search similar papers,  i.e. papers which touch on the 1 or 3 KEY AREAS OF THE PAPER IMPORTANT
Rules:
1. TITLE: Exact paper title
2. Each CORE_METHODOLOGY should be:,A basic technique/method + scope qualifier,General enough to catch variations,Specific enough to stay relevant
3. A very short sentence which takes the topics presented in the paper and upon text search would return papers with an interesting slant on the orignal topics
4. Maximum of 3 core methodologies
5. Format each as: [basic_technique] + [scope_term(s)]
6. Random should be tipic unrelated to scientific paper

Good examples:
For a neural network paper:"Gradient optimization for adversarial attacks","Cross validation for network robustness","Model compression for edge devices"

For a genomics paper:"Clustering for gene expression","Sequence alignment for protein families","Feature selection for biomarkers"

Bad examples:
Too general: "Neural networks", "Clustering algorithms","Statistical analysis"

Too specific: "Three-layer perceptron with ReLU for MNIST","K-means clustering of 50000 gene sequences","Chi-square analysis of patient outcomes"

Output format:
TITLE: [exact paper title];
CORE_METHODOLOGIES: [method1, method2, method3];
CONCEPTUAL_ANGLES: [angle1, angle2, angle3s];
RANDOM: [ 'mountain climbing','abstract painting','quantum dice' ]
Note: Keep it simple - basic technique plus scope, nothing more. For interesting slant, keep it simple main ideas of paper + something more abstract then core method
ANY DEVIATION IN THIS FORMAT WILL LEAD TO ERRORS IN A DETERMINISTIC SYSTEM
DO NOT ALTER THE OUTPUT FORMAT WITH NEW LINES DASHES OR ANY OTHER CHARACTER STCK STRICTLY TO OUTLINED FORMAT


"""



claudeInstruction_extractAllInfo = '''Given a scientific paper, perform a DETERMINISTIC analysis by following these strict steps in order. Your output must be EXACTLY consistent across multiple runs.

Output must be in this exact format with two sections separated by a blank line:

METADATA:
TITLE: [paper title];
AUTHORS: [comma-separated list of authors];
YEAR: [publication year];
ABSTRACT: [paper abstract];
CITATION_COUNT: [number of citations if available, else 0];
REFERENCE_COUNT: [number of references if available, else 0];

ANALYSIS:
CORE_CONCEPTS: [only concepts from abstract, comma-separated];
CORE_METHODOLOGIES: [only described methods, comma-separated];
RELATED_METHODOLOGIES: [standard field methods, comma-separated];
ABSTRACT_CONCEPTS: [max 3 principles, comma-separated];
CROSS_DOMAIN: [max 3 fields, comma-separated];
THEORETICAL_FOUNDATIONS: [max 3 theories, comma-separated];
ANALOGOUS_PROBLEMS: [max 3 problems, comma-separated];

Rules for deterministic output:
- Use ONLY information explicitly stated in the paper
- Limit each analysis section to 3 items maximum
- Always order items alphabetically within each section
- Use consistent, standardized terminology
- No explanatory text or variations
- No optional or variable content
- Strict semicolon and comma usage
- All sections must be present even if empty (use empty string)
Any deviation from this format or inconsistency across runs is an error. ANY DEVIATION AT ALL IN THE DETERMINISTIC ANSWERS IS AN ERROR AND WILL RESULT IN A NON FUNCITONAL SYSTEM.
'''

current_dir = Path(__file__).parent
similarity_root = current_dir.parent
env_path = similarity_root / '.env.txt'

upload_bp = Blueprint('upload', __name__)
load_dotenv(env_path)  # Load environment variables from .en

api_key_semantic = os.getenv('SEMANTIC_API_KEY')
api_key_claude = os.getenv('HAIKU_API_KEY')

upload_bp = Blueprint('upload', __name__)
# Create an upload folder for temporary file storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@upload_bp.route('/process-pdf', methods=['POST'])
def process_pdf_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    functionName = request.form.get('functionName')
    pdfName = request.form.get('pdfPath')
    print(pdfName)
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Initialize our processor
        processor = PDFProcessor()
        semanticScholar = SemanticScholar()
        cache = SearchTermCache()
        
        # Will probably have to change to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        if functionName == 'extractSeedPaperInfo':
            entireFuncionTime = time.time()
            try:
                entirePDFText = processor._extract_text(filepath)
                if len(entirePDFText) > 10024:
                    entirePDFText = entirePDFText[:8000]
                
                # Extract title and initial analysis
                pdfTitle_claude = processor.ask_claude(text=entirePDFText, 
                                                     instruction=claudeInstruction_extractTitleMethodInfo,
                                                     api_key=api_key_claude)
                pdfInfo = pdfTitle_claude.content[0].text
                print(f"Extracted info: {pdfInfo}")
                pdfInfoStruct = processor.parse_paper_info(pdfInfo)
                
                # Try Semantic Scholar first
                startTime = time.time()
                print('The title is',pdfInfoStruct['title'])
                semanticScholarPaperInfo = semanticScholar.return_info_by_title(pdfInfoStruct['title'], 
                                                                              api_key_semantic)
                endTime = time.time()
                print(f"Time taken for semantic scholar search: {endTime - startTime} seconds")
                if semanticScholarPaperInfo:
                    print("Using Semantic Scholar data")
                    # Use Semantic Scholar data if available
                    result = {
                        'title': pdfInfoStruct['title'],
                        'paper_info': {  
                            'authors': semanticScholarPaperInfo['authors'],
                            'abstract': semanticScholarPaperInfo['abstract'],
                            'year': semanticScholarPaperInfo['year'],
                            'citation_count': semanticScholarPaperInfo['citation_count'],
                            'reference_count': semanticScholarPaperInfo['reference_count'],
                            'citations': semanticScholarPaperInfo['citations'],
                            'references': semanticScholarPaperInfo['references']
                        },
                        'abstract_info': {
                            'core_concepts': pdfInfoStruct['core_methodologies'],
                            'conceptual_angles':pdfInfoStruct['conceptual_angles'],
                            'random':pdfInfoStruct['random']
                        }
                    }
                else:
                    print("No semantic scholar data")
                    # If no Semantic Scholar data, use Haiku analysis
                    entirePDFText = processor._extract_text(filepath)  # Get full text again if needed
                    seedPaperInfo = processor.ask_claude(text=entirePDFText, 
                                                       instruction=claudeInstruction_extractAllInfo, 
                                                       api_key=api_key_claude)
                    haikuResults = processor.parse_haiku_output(seedPaperInfo.content[0].text)
                    
                    result = {
                        'title': pdfInfoStruct['title'],
                        'paper_info': {  # Use Haiku's extracted info
                            'authors': haikuResults['semantic_scholar_info']['authors'],
                            'abstract': haikuResults['semantic_scholar_info']['abstract'],
                            'year': haikuResults['semantic_scholar_info']['year'],
                            'citation_count': haikuResults['semantic_scholar_info']['citation_count'],
                            'reference_count': haikuResults['semantic_scholar_info']['reference_count'],
                            'citations': [],  # Empty list as Haiku can't get actual citations
                            'references': []  # Empty list as Haiku can't get actual references
                        },
                        'abstract_info': {
                            'core_concepts': pdfInfoStruct['core_concepts'],
                            'core_methodologies': pdfInfoStruct['core_methodologies'],
                            'related_methodologies': pdfInfoStruct['related_methodologies'],
                            'key_findings': pdfInfoStruct['key_findings'],
                            'subject_tags': pdfInfoStruct['subject_tags'],
                        }
                    }
                
                papersReturnedThroughSearch = []
                
                # Now based on the abstract info extracted from the paper we should search semantic scholar for similar papers
                # Loop below searches semantic scholar using each of the core concepts and returns 5 papers for each of those concepts
                # if 3 concepts than 15 papers
                # We store 'search_type' so we can indicate to the user what type of search was done to get this paper
                startTime = time.time()
                semanticScholar = SemanticScholar()
                cache = SearchTermCache()
                search_terms = []
                 # Add core methodologies with high weight since they're specific
                if 'core_methodologies' in pdfInfoStruct and pdfInfoStruct['core_methodologies']:
                 for methodology in pdfInfoStruct['core_methodologies']:
                   search_terms.append({
               'term': methodology,
               'type': 'core_methodology',
               'weight': 1.0 # High weight for specific methodologies
           })
                if 'conceptual_angles' in pdfInfoStruct and pdfInfoStruct['conceptual_angles']:
                 for conceptualAngle in pdfInfoStruct['conceptual_angles']:
                   search_terms.append({
               'term': conceptualAngle,
               'type': 'conceptual_angles',
               'weight': 1.0 # High weight for specific methodologies
           })
                if 'random' in pdfInfoStruct and pdfInfoStruct['random']:
                 for randomSubject in pdfInfoStruct['random']:
                   search_terms.append({
               'term': randomSubject,
               'type': 'random',
               'weight': 1.0 # High weight for specific methodologies
           })

                # Do all searches in parallel
                startTime = time.time()
                print('search terms are')
                print(search_terms)
                papersReturnedThroughSearch = semanticScholar.search_papers_parallel(search_terms, api_key_semantic)
                endTime = time.time()
                print(f"Time taken for searching using core techniques: {endTime - startTime} seconds")          
                metricsCalculator = MetricsCalculator()

                # Get embedding for seed paper
                seed_abstract = semanticScholarPaperInfo['abstract']
                seed_embedding = metricsCalculator.get_scibert_embedding(seed_abstract, tokenizer, model)
                semanticScholarPaperInfo['scibert'] = seed_embedding.tolist()
        
                seedPaper = {
                    'search_type': 'seed_paper',
                    'paper_info': semanticScholarPaperInfo
                }
                # Compare seed paper against all papers returned through search
                startTime = time.time()
                print("Comparing papers...")
                similarityResults  = compare_papers(seedPaper, papersReturnedThroughSearch)
                print("Finished comparing papers")
                endTime = time.time()
                print(f"Time taken for comparison: {endTime - startTime} seconds")
                
                # From returned papers and their simlarity score, get only relatively similar papers
                startTime = time.time()
                relativelySimilarPapers = metricsCalculator.apply_source_weights(similarityResults['compared_papers'])
                relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(relativelySimilarPapers['compared_papers'])
                endTime = time.time()
                print(f"Time taken for filtering similar papers: {endTime - startTime} seconds")
        
        
                # Remove the 'scibert' attribute from relatively similar papers
                for paper in relativelySimilarPapers:
                    if 'scibert' in paper['paper_info']:
                        del paper['paper_info']['scibert']
                # Clean up and return
                os.remove(filepath)
                result['seed_paper'] = seedPaper
                result['similarity_results'] = relativelySimilarPapers
                result['test'] = similarityResults
                finishingTime = time.time()
                print(f"Entire function took: {finishingTime - entireFuncionTime} seconds")
                 # Write the result to a file named after the seed paper
                seed_paper_title = result['title'].replace(' ', '_')
                output_filename = f"{seed_paper_title}_result.json"
                with open(output_filename, 'w') as outfile:
                    json.dump(result, outfile, indent=4)
                        
                return jsonify(result), 200
                   
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                print(f"Detailed error: {error_message}")  # Enhanced error logging
                
                # Clean up if file exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return jsonify({
                    'error': error_message,
                    'status': 'error'
                }), 500

    except Exception as e:
        # Log the full error with traceback
        import traceback
        error_msg = f"Error processing PDF: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  
        
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 500

# Add the project root directory to Python path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    global tokenizer, model
    try:
        # Try to load from local cache only
        print("Loading SciBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', local_files_only=True)
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', local_files_only=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Model moved to GPU")
    except Exception as e:
        print(f"Error loading from cache, downloading model: {e}")
        # If not in cache, download it
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("SciBERT model loaded successfully")
    app.register_blueprint(upload_bp)
    return app

# This function should probably be moved to another file????//
def compare_papers(seed_paper, papers_returned_through_search):
    try:
        inference = ModelInferenceThree(
            model_path="modelFolder/standardModel-3-37k.pth",
        )
        metricCalculator = MetricsCalculator()
        
        # Get all metrics in parallel
        metrics_list = metricCalculator.process_papers_parallel(
            seed_paper, 
            papers_returned_through_search,
            tokenizer, 
            model
        )
        
        compared_papers = []
        # Process the results
        for paper, metrics in zip(papers_returned_through_search, metrics_list):
            # Prepare shared data dictionary
            shared_data = {
                'reference_count': metrics['shared_reference_count'],
                'reference_cosine': metrics['reference_cosine'],
                'citation_count': metrics['shared_citation_count'],
                'citation_cosine': metrics['citation_cosine'],
                'author_count': metrics['shared_author_count'],
                'abstract_cosine': metrics['abstract_cosine']
            }
            
            try:
                similarity = inference.predict_similarity(
                    paper1_SciBert=seed_paper['paper_info'].get('scibert', []),
                    paper2_SciBert=paper['paper_info'].get('scibert', []),
                    shared_data=shared_data
                )
            except Exception as e:
                logging.error(f"Error in similarity prediction: {str(e)}")
                similarity = 0.0
            
            compared_paper = {
                'source_info': paper.get('source_info', {}),
                'paper_info': paper.get('paper_info', {}),
                'similarity_score': float(similarity),
                'comparison_metrics': metrics
            }
            compared_papers.append(compared_paper)

        return {
            'seed_paper': {
                'paper_info': seed_paper
            },
            'compared_papers': compared_papers
        }
        
    except Exception as e:
        logging.error(f"Error occurred during paper comparison: {str(e)}")
        raise

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
