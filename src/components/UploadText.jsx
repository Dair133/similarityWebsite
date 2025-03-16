import React, { useState } from 'react';

function UploadText({ onSearchSubmit, processing }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [dots, setDots] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSearchSubmit(searchQuery);
    }
  };

  const handleInputChange = (e) => {
    setSearchQuery(e.target.value);
  };

  // Example queries for user reference
  const exampleQueries = [
    "I'm looking for papers about climate change prediction models that incorporate economic factors",
    "Find me recent research on how gut microbiome affects mental health, especially studies using machine learning",
    "I need papers exploring quantum computing applications in cryptography from the last 3 years"
  ];
  const styles = {
    container: {
      backgroundColor: 'white',
      borderRadius: '8px',
      padding: '2rem',
      width: '90%',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
    },
    title: {
      fontSize: '24px',
      color: '#333',
      marginBottom: '1.5rem'
    },
    form: {
      width: '100%'
    },
    inputContainer: {
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      marginBottom: '1rem'
    },
    textInput: {
      padding: '12px',
      fontSize: '16px',
      borderRadius: '4px',
      border: '1px solid #ccc',
      width: '100%',
      boxSizing: 'border-box'
    },
    button: {
      backgroundColor: '#007bff',
      color: 'white',
      padding: '12px 20px',
      border: 'none',
      borderRadius: '4px',
      fontSize: '16px',
      cursor: 'pointer',
      width: '100%',
      transition: 'background-color 0.3s'
    },
    disabledButton: {
      backgroundColor: '#cccccc',
      cursor: 'not-allowed'
    },
    examplesSection: {
      marginTop: '1.5rem',
      padding: '1rem',
      backgroundColor: '#f9f9f9',
      borderRadius: '4px',
      border: '1px solid #eee'
    },
    examplesTitle: {
      fontSize: '16px',
      color: '#555',
      marginBottom: '0.5rem',
      fontWeight: 'bold'
    },
    examplesList: {
      margin: '0',
      padding: '0 0 0 20px'
    },
    exampleItem: {
      margin: '0.5rem 0',
      color: '#007bff',
      cursor: 'pointer'
    },
    note: {
      fontSize: '14px',
      color: '#666',
      marginTop: '1rem',
      fontStyle: 'italic'
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Semantic Paper Search</h2>
      
      <form style={styles.form} onSubmit={handleSubmit}>
        <div style={styles.inputContainer}>
          <input
            type="text"
            placeholder="Type your scientific paper search query here..."
            value={searchQuery}
            onChange={handleInputChange}
            style={styles.textInput}
            disabled={processing}
          />
          
          <button 
            type="submit" 
            style={{
              ...styles.button,
              ...(processing ? styles.disabledButton : {})
            }}
            disabled={processing || !searchQuery.trim()}
          >
            {processing ? `Processing${dots}` : 'Search for Papers'}
          </button>
        </div>
      </form>
      
      <div style={styles.examplesSection}>
        <p style={styles.examplesTitle}>Example searches:</p>
        <ul style={styles.examplesList}>
          {exampleQueries.map((query, index) => (
            <li 
              key={index} 
              style={styles.exampleItem}
              onClick={() => setSearchQuery(query)}
            >
              {query}
            </li>
          ))}
        </ul>
        
        <p style={styles.note}>
          Feel free to use natural language to describe what you're looking for. Include context about your research interests, 
          specific questions you have, or relationships between topics you want to explore. The more detail you provide about 
          your interests, the better the results will be.
        </p>
      </div>
    </div>
  );
}

export default UploadText;