# backend/routes/upload_routes.py
# APP.py script
from flask import Blueprint, request, jsonify
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
claudeInstruction_extractInfo = """
Please analyze this scientific paper and extract information in EXACTLY the following format, with NO deviation:

TITLE:'[Insert the main research title - not journal name, section headers, or running headers]';
CORE_CONCEPTS:[concept1],[concept2],[concept3];
CORE_METHODOLOGIES:[method1],[method2],[method3];
RELATED_METHODOLOGIES:[method1],[method2],[method3];

Important formatting rules:
1. Everything must be on one line
2. Use semicolons (;) to separate major sections
3. Use commas (,) to separate items within sections
4. Do not use bullet points
5. Do not use line breaks
6. Include square brackets around each individual item
7. No spaces between commas and next item
8. Use capital letters at beginning of each , i.e. Medicine,Biology, this info will be printed and shoul look presentable
Example of correct format:
TITLE:'Ozone therapy mitigates parthanatos after ischemic stroke';CORE_CONCEPTS:[ischemic stroke],[ozone therapy],[neural recovery];CORE_METHODOLOGIES:[brain imaging],[blood analysis],[behavioral testing];RELATED_METHODOLOGIES:[drug delivery],[tissue oxygenation],[neural monitoring];

For the title:
- Extract only the primary research title
- Exclude journal names, section headers, or running headers
- If no title is found, return 'Title not found'

For concepts and methodologies:
- List in order of importance
- Be specific but concise
- Use consistent terminology
- Include only items explicitly mentioned or directly implied in the paper

Remember: The exact format is critical for automated processing. Any deviation from this format will cause errors in the system.
"""


# ALL API KEYS HANDLED BELOW

# Get the current file's directory
current_dir = Path(__file__).parent
# Navigate up to similarity folder
similarity_root = current_dir.parent
env_path = similarity_root / '.env.txt'


upload_bp = Blueprint('upload', __name__)
load_dotenv(env_path)  # Load environment variables from .en


api_key_semantic= os.getenv('SEMANTIC_API_KEY')
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
        # Will probably have ti chaneg to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        if functionName == 'extractSeedPaperInfo':
            try:
                entirePDFText = processor._extract_text(filepath)
                if len(entirePDFText) > 10024:
                    entirePDFText = entirePDFText[:8000]
                
                # Get Claude's response - this will now be a string
                pdfTitle_claude = processor.ask_claude(
                    text=entirePDFText,
                    instruction=claudeInstruction_extractInfo,
                    api_key=api_key_claude
                )
                print('being returned from claude is',pdfTitle_claude)
                # Extract just the title text from the textblock and clean it up
                pdfInfo = pdfTitle_claude.content[0].text
            
                pdfInfoStruct = processor.parse_paper_info(pdfInfo)
                
                # Now pass the clean title to semantic scholar
                semanticScholarPaperInfo = semanticScholar.return_info_by_title(pdfInfoStruct['title'], api_key_semantic)
                # Now search semantic scholar for that paper. If no result returned then
                # we will form the info we need ourselves using haiku.
                
                result = {
                'title': pdfInfoStruct['title'],
                'semantic_scholar_info': {
                'title': semanticScholarPaperInfo['Title'],
                'authors': semanticScholarPaperInfo['Authors'], 
                'abstract': semanticScholarPaperInfo['Abstract'],
                'year': semanticScholarPaperInfo['Year'],
                'citation_count': semanticScholarPaperInfo['Citation_Count'],
                'reference_count': semanticScholarPaperInfo['Reference_Count'],
                'citations': semanticScholarPaperInfo['Citations'],
                'references': semanticScholarPaperInfo['References']
                    },
                'abstract_info':{
                    'core_concepts': pdfInfoStruct['core_concepts'],
                    'core_methodologies': pdfInfoStruct['core_methodologies'],
                    'related_methodologies': pdfInfoStruct['related_methodologies']
                }
                }
        
                # Clean up and return
                os.remove(filepath)
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
        print(error_msg)  # This will show in your Flask console
        
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
    
    app.register_blueprint(upload_bp)
    
    return app



@upload_bp.route('/comparePapers', methods=['POST'])
def compare_papers_route():
    try:
        inference = ModelInference("modelFolder/best_model_fold_4.pth")  
        
        # Test data with complete 768-dimensional vectors
        paper1_scibert = [0.7303557395935059, -0.19036521017551422, -0.20561213791370392]  
        paper2_scibert = [0.1599576771259308, 0.0030540116131305695, -0.7980160713195801]  
        
        # Pad vectors to 768 dimensions with zeros for testing
        paper1_scibert.extend([0.0] * (768 - len(paper1_scibert)))
        paper2_scibert.extend([0.0] * (768 - len(paper2_scibert)))
        
        data = {
            "paper1_Citation_Count": 59,
            "paper1_Reference_Count": 82,
            "paper1_SciBert": paper1_scibert,
            "paper2_Citation_Count": 161,
            "paper2_Reference_Count": 0,
            "paper2_SciBert": paper2_scibert,
            "shared_author_count": 0,
            "shared_reference_count": 3,
            "shared_citation_count": 3,
            "reference_cosine": 0.014613731,
            "citation_cosine": 0.725575944,
            "abstract_cosine": 0.5472629
        }

        # Add debugging prints
        print("Starting prediction...")
        print(f"SciBERT dimensions: {len(data['paper1_SciBert'])}, {len(data['paper2_SciBert'])}")
        
        similarity = inference.predict_similarity(
            paper1_Citation_Count=data['paper1_Citation_Count'],
            paper1_Reference_Count=data['paper1_Reference_Count'],
            paper1_SciBert=data['paper1_SciBert'],
            paper2_Citation_Count=data['paper2_Citation_Count'],
            paper2_Reference_Count=data['paper2_Reference_Count'],
            paper2_SciBert=data['paper2_SciBert'],
            shared_author_count=data['shared_author_count'],
            shared_reference_count=data['shared_reference_count'],
            shared_citation_count=data['shared_citation_count'],
            reference_cosine=data['reference_cosine'],
            citation_cosine=data['citation_cosine'],
            abstract_cosine=data['abstract_cosine']
        )
        
        print('Similarity score:', similarity)
        return jsonify({'similarityScore': float(similarity)}), 200
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Add this for debugging
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app = create_app()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
