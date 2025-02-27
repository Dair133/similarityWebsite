import os
import hashlib
import json
import re
from typing import Dict, Optional, Any


class SearchTermCache:
    def __init__(self, cache_dir: str = "search_term_cache"):
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.cache_file_path = os.path.join(cache_dir, "cache.json")
        
        
        if not os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'w') as f:
                json.dump({}, f)


    def createCacheFile(self, paperCacheTitle, pdfDataToCache):
     try:
        # Create the directory structure for the cache file if needed
        directory = os.path.dirname(self.cache_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Create a new cache file with the paper data as the first entry
        initial_cache = {paperCacheTitle: pdfDataToCache}
        
        # Write the cache file with the paper data
        with open(self.cache_file_path, 'w') as f:
            json.dump(initial_cache, f, indent=2)
        
        return True
    
     except Exception as e:
        print(f"Error creating cache file: {str(e)}")
        return False
    
    def cacheCheck(self,pdfTitle, pdfDataToCache):
        currentCacheLength = self.count_cache_entries()
        print("Checking cache, there are currently "+currentCacheLength+"entries")
        # If json file has a entry under 
        paperCacheTitle = self.generatePaperId(pdfTitle)
        
        try:
            with open(self.cache_file_path,'r') as f:
                cache_data = json.load(f)
        
            if(paperCacheTitle in cache_data):
                return cache_data[paperCacheTitle]
            else:
            # If error when reading thena ssume that paper does not exist
                self.addPaperCache(paperCacheTitle, pdfDataToCache)
                return False
        except json.JSONDecodeError:
            print('Error when searching for cache - JSON Decode error')
            return False
        except FileNotFoundError:
            print('Cache File not found - creating new cache file')
            self.createCacheFile(paperCacheTitle, pdfDataToCache)
            return False
             
    
    
    def generatePaperId(self,pdfTitle):
        cleaned_title = pdfTitle.strip().lower()
        return hashlib.md5(cleaned_title.encode('utf-8')).hexdigest()
    
    def addPaperCache(self,pdfHashTitle,dataToStore):
                # Make sure our cache file exists
        if not os.path.exists(self.cache_file_path):
            # Create an empty dictionary in the cache file
            with open(self.cache_file_path, 'w') as f:
                json.dump({}, f)
                
                
        try:
            with open(self.cache_file_path, 'r')as f:
                cache_data = json.load(f)
        except json.JSONDecodeError:
            print('Error in parsing cache file, please review')
            cache_data = {}
            
        cache_data[pdfHashTitle] = dataToStore
        
        try:
            with open(self.cache_file_path,'w') as f:
                json.dump(cache_data,f,indent = 2)
        except:
            print('Error when saving new paper to cache, please review')

            
            
    def count_cache_entries(self):
     try:
        # Check if the cache file exists
        if not os.path.exists(self.cache_file_path):
            return 0
            
        # Open and read the cache file
        with open(self.cache_file_path, 'r') as f:
            cache_data = json.load(f)
            
        # Return the number of entries
        return len(cache_data)
        
     except json.JSONDecodeError:
        # This occurs if the file exists but contains invalid JSON
        print("Error: Cache file contains invalid JSON format")
        return -1
        
     except Exception as e:
        # Catch any other unexpected errors
        print(f"Error counting cache entries: {str(e)}")
        return -1
            
        
        
        