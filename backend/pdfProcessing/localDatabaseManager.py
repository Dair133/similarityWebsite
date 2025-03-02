import os
import hashlib
import json
import re
from typing import Dict, Optional, Any
import pandas as pd
from pathlib import Path


class LocalDatabaseManager:
        # Every local excel file will have the following info about the paper
            # SciBert
            # Title
            # Abstract
            # References
            # Citations
            # Authors
    def load_poison_pill_papers(self, file_name="poison_pill_papers_With_SciBert.xlsx"):
     """
    Load poison pill papers from Excel file and format them to match the existing data structure.
    The Excel file should be located in a 'poisonPill' folder in the same directory as this script.
    """
     try:
        import pandas as pd
        from pathlib import Path
        import os
        
        # Get the directory where the LocalDatabaseManager class is defined
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for the file in the poisonPill folder
        poison_pill_dir = os.path.join(module_dir, "poisonPill")
        file_path = os.path.join(poison_pill_dir, file_name)
        
        # Check if file exists
        if not Path(file_path).exists():
            print(f"Warning: Poison pill papers file not found at {file_path}")
            return []
        
        # Read Excel file
        print(f"Loading poison pill papers from {file_path}")
        df = pd.read_excel(file_path)
        
        # Convert DataFrame to list of dictionaries that match our expected structure
        poison_pill_papers = []
        
        for _, row in df.iterrows():
            try:
                # Handle the year specifically to avoid NaN
                year_value = row.get('Year')
                if pd.isna(year_value):
                    year_value = "Unknown"
                elif isinstance(year_value, (int, float)):
                    # Convert to int if it's a valid number
                    try:
                        year_value = int(year_value)
                    except:
                        year_value = "Unknown"
                else:
                    year_value = str(year_value)
                
                # Create paper structure matching what's expected by the comparison function
                paper = {
                    'source_info': {
                        'search_term': 'poisonPill',
                        'search_type': 'poisonPill'
                    },
                    'paper_info': {
                        'title': str(row.get('Title', 'Unknown Title')),
                        'abstract': str(row.get('Abstract', '')),
                        'year': year_value,
                        'authors': str(row.get('Authors', '')),
                        'citations': str(row.get('Citations', '')),
                        'references': str(row.get('References', '')),
                        'citation_count': 0,  # Default values
                        'reference_count': 0
                    },
                    'comparison_metrics': {
                        'shared_reference_count': 0,
                        'shared_citation_count': 0, 
                        'shared_author_count': 0
                    }
                }
                
                # Handle SciBert embedding if available
                if 'SciBert' in row and not pd.isna(row['SciBert']):
                    # Convert string representation of list to actual list
                    scibert_str = row['SciBert']
                    if isinstance(scibert_str, str):
                        import json
                        # Try to parse as JSON
                        try:
                            scibert = json.loads(scibert_str)
                            paper['paper_info']['scibert'] = scibert
                        except json.JSONDecodeError:
                            # If not valid JSON, try eval (less safe but might work for array literals)
                            try:
                                scibert = eval(scibert_str)
                                paper['paper_info']['scibert'] = scibert
                            except:
                                print(f"Could not parse SciBert for {row.get('Title')}")
                
                poison_pill_papers.append(paper)
                
            except Exception as e:
                print(f"Error processing row in poison pill papers file: {e}")
                continue
        
        print(f"Loaded {len(poison_pill_papers)} poison pill papers")
        return poison_pill_papers
        
     except Exception as e:
        print(f"Error loading poison pill papers: {e}")
        return []
        
        # Ok we need a funciton which simply returns the papers from the poison_pill_papers_With_SciBert.xlsx in the correct data structure
        
        
        # TO BE IMPLEMENTED
        # Dont do this yet but we are gonna need some form of pre sampling, so we DONT have to compare the seed paper to every paper int he local db
        # Im thinking we compare seed paper to 5 papers from every main topic, get average similarity per main topic
        # THen compare seed paper with only papers from main topic that have highest similarity
        