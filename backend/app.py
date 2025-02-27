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
from modelFolder.modelRunners.standardModelRunner32k3 import ModelInference
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
# Think of search paper string as seed
claudeInstruction_extractTitleMethodInfo = """
# Scientific Paper Analysis System

You are a professional scientific literature analyst specializing in precise terminology extraction. Your extracted terms will be used directly in an automated academic search pipeline that feeds into a machine learning model. This pipeline searches Semantic Scholar using your extracted terms to find papers similar to a "seed paper," and these results are then evaluated by an ML model that assesses their similarity to the original paper.

## Core Extraction Task
Extract the exact paper title, identify 3 core methodologies, and create 3 conceptual angles that will yield effective literature search results. The quality of your extracted terms directly impacts the effectiveness of the entire ML pipeline.

## Output Format (CRITICAL - MUST BE FOLLOWED EXACTLY)
TITLE: [exact paper title];
CORE_METHODOLOGIES: [method1, method2, method3];
CONCEPTUAL_ANGLES: [angle1, angle2, angle3];

## CORE_METHODOLOGIES Guidelines
- Create exact search strings that would return the MOST SIMILAR papers to the source when searched on Semantic Scholar
- Each methodology must combine a fundamental technique with precise scope-defining terms
- The string should be specific enough that searching with it would NOT return papers from unrelated domains
- For each term, ask: "Would this search term potentially return completely unrelated papers?" If yes, make it more specific
- Avoid overly general terms like "neural networks" or "optimization" without proper qualification
- These terms must identify papers that use highly similar approaches to the seed paper

## CONCEPTUAL_ANGLES Guidelines
- Create search strings that would return papers with a SLIGHT CONCEPTUAL SLANT from the source
- These should capture related approaches that explore the same problem from different angles
- Each angle should maintain relevance to the core topic while introducing a novel perspective
- Strike a careful balance between similarity and novelty - too similar will duplicate core methodologies, too different will return irrelevant papers
- The ML model can distinguish broadly between similar and dissimilar papers, so these terms should find papers that are related but offer new insights

## Examples of Effective Search Strings:
CORE: "Siamese networks for document similarity" (returns highly similar papers)
ANGLE: "Cross-modal embeddings for text retrieval" (returns papers with a related but different approach)

## FORMAT WARNING (EXTREMELY IMPORTANT)
The output format MUST be followed EXACTLY as specified. Any deviation will cause the entire ML pipeline to fail. This includes:
- Using the exact section headers shown
- Including the square brackets around each term
- Using semicolons as separators exactly as shown
- Not adding any extra characters, line breaks, or explanations
- Providing exactly 3 items for each category

This output is programmatically parsed and any formatting change will break the automated system.
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
env_path = similarity_root / '.env'

upload_bp = Blueprint('upload', __name__)
load_dotenv(env_path)  # Load environment variables from .en
api_key_gemini = os.getenv('GEMINI_API_KEY')
api_key_semantic = os.getenv('SEMANTIC_API_KEY')
api_key_claude = os.getenv('HAIKU_API_KEY')
api_key_deepseek = os.getenv('DEEPSEEK_API_KEY')
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
        print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
        print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
        # Initialize our processor
        processor = PDFProcessor()
        semanticScholar = SemanticScholar()
        cache = SearchTermCache()
        metricsCalculator = MetricsCalculator()
        # processor.ask_gemini(api_key_gemini)
        #processor.ask_deepseek(api_key_deepseek)
        # Will probably have to change to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        if functionName == 'extractSeedPaperInfo':
            entireFuncionTime = time.time()
            try:
                entirePDFText = processor._extract_text(filepath)
                if len(entirePDFText) > 10024:
                    entirePDFText = entirePDFText[:8000]
                
                # Extract title and initial analysis
                pdfTitle_claude = processor.ask_claude(pdfText=entirePDFText, 
                                                     systemInstructions=claudeInstruction_extractTitleMethodInfo,
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
                
                
                # Get embedding for seed paper
                seed_abstract = semanticScholarPaperInfo['abstract']
                seed_embedding = metricsCalculator.get_scibert_embedding(seed_abstract, tokenizer, model)
                semanticScholarPaperInfo['scibert'] = seed_embedding.tolist()
                
                # Here we will calculate shared references, citations and cosine
                seedReferenceList = semanticScholarPaperInfo['references']
                parsedSeedReferenceList = metricsCalculator.parse_attribute_list(seedReferenceList,';')
                
                seedCitationList = semanticScholarPaperInfo['citations']
                parsedSeedCitationList = metricsCalculator.parse_attribute_list(seedCitationList,';')
                
                seedAuthorList = semanticScholarPaperInfo['authors']
                print('SEed author list below')
                print(seedAuthorList)
                parsedSeedAuthorLust = metricsCalculator.parse_attribute_list(seedAuthorList, ',')
                
                
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
                if len(seedAuthorList) > 0:
                 for author in parsedSeedAuthorLust:
                    search_terms.append({
                    'term':author,
                    'type':'author',
                    'weight':1.0
                })

                # Do all searches in parallel
                startTime = time.time()
                print('search terms are')
                print(search_terms)
                papersReturnedThroughSearch = semanticScholar.search_papers_parallel(search_terms, api_key_semantic)
                endTime = time.time()
                print(f"Time taken for searching using core techniques: {endTime - startTime} seconds")          




                
        
                seedPaper = {
                    'search_type': 'seed_paper',
                    'paper_info': semanticScholarPaperInfo
                }
                # Compare seed paper against all papers returned through search
                startTime = time.time()
                print("Comparing papers...")
                similarityResults  = compare_papers(seedPaper, papersReturnedThroughSearch)

                for paper in similarityResults['compared_papers']:
                    referenceList = paper['paper_info'].get('references', [])
                    sharedReferenceCount = metricsCalculator.compareAttribute(parsedSeedReferenceList, referenceList)
                    paper['comparison_metrics']['shared_reference_count'] = sharedReferenceCount
                        
                    citationList = paper['paper_info'].get('citations', [])
                    sharedCitationCount = metricsCalculator.compareAttribute(parsedSeedCitationList,citationList)
                    paper['comparison_metrics']['shared_citation_count'] = sharedCitationCount
                    
                    authorList = paper['paper_info'].get('authors', [])
                    sharedAuthorCount = metricsCalculator.compareAttribute(parsedSeedAuthorLust,authorList)
                    paper['comparison_metrics']['shared_author_count'] = sharedAuthorCount
                    
                    if sharedReferenceCount > 0:
                        print('Found authors or citaitons or references greater than 0',sharedAuthorCount,sharedCitationCount,sharedReferenceCount)
                    

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
        inference = ModelInference(
            model_path="modelFolder/standardModel-3-32k-NoLeakage.pth",
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
            # Updated shared data to only include the three features used by new model
            shared_data = {
                'shared_references': metrics['shared_reference_count'],
                'shared_citations': metrics['shared_citation_count'],
                'shared_authors': metrics['shared_author_count']
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
            'compared_papers': compared_papers,
        }
        
    except Exception as e:
        logging.error(f"Error occurred during paper comparison: {str(e)}")
        # Return fake results in case of failure
        fake_results = [
            {
                'source_info': {},
                'paper_info': {},
                'similarity_score': 0.5,
                'comparison_metrics': {}
            }
        ]
        return {
            'seed_paper': {
                'paper_info': seed_paper
            },
            'compared_papers': fake_results,
            'search_terms': 'Fake Search'
        }

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
