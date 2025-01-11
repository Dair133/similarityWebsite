import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  //  Called initially with a value of null i.e. sets the initial state of results to null
  // Below format is a special syntax in React that allows you to destructure an array into multiple variables
  // The first element of the array is the state variable, and the second element is a function that allows you to update the state variable
  // it is the 'useState' hook which tells the array to be destructured into two variables - x and setX
  const [results, setPaperData] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      handleUpload(file);  // Automatically upload when file is selected
    } else {
      alert('Please upload a PDF file');
    }
  };

  const handleUpload = async (file) => {
    try {
      setProcessing(true);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('functionName', 'extractSeedPaperInfo');
      const response = await fetch('http://localhost:5000/process-pdf', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();

      console.log(data)
      // setsPaper data, triggering the useState hook to update the results state variable
      setPaperData(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to process the file');
    } finally {
      setProcessing(false);
    }
  };

  const styles = {
    container: {
      minHeight: '100vh',
      width: '100%',
      backgroundColor: 'black',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    },
    whiteBox: {
      backgroundColor: 'white',
      borderRadius: '8px',
      padding: '40px',
      width: '100%',
      maxWidth: '500px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '20px'
    },
    title: {
      fontSize: '24px',
      color: '#333',
      marginBottom: '20px'
    },
    uploadArea: {
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center'
    },
    input: {
      width: '100%',
      padding: '10px',
      border: '2px dashed #ccc',
      borderRadius: '4px',
      cursor: 'pointer'
    },
    fileInfo: {
      marginTop: '10px',
      color: '#666',
      fontSize: '14px'
    },
    results: {
      marginTop: '20px',
      width: '100%',
      textAlign: 'left'
    }
  };

  const renderSemanticScholarInfo = (info) => {
    return (
      <div>
        {Object.entries(info).map(([key, value]) => (
          <div key={key}>
            <strong>{key}:</strong> {JSON.stringify(value, null, 2)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.whiteBox}>
        <h2 style={styles.title}>Upload PDF</h2>
        <div style={styles.uploadArea}>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            style={styles.input}
          />
          {selectedFile && (
            <div style={styles.fileInfo}>
              Selected file: {selectedFile.name}
            </div>
          )}
          {processing && <div>Processing...</div>}
          {results && (
            <div style={styles.results}>
              <h3>Semantic Scholar Info:</h3>
              {renderSemanticScholarInfo(results.semantic_scholar_info)}
              <h3>Semantic Scholar Abstract Info:</h3>
              {renderSemanticScholarInfo(results.abstract_info)}
              <h3>Similar Papers</h3>
              <ul>
                {results.similarity_results.map((paper, index) => (
                  <li key={index}>
                    <strong>Title: {paper.title}</strong>Paper ID: {paper.id} - Similarity Score: {paper.similarity_score}<br></br>
                    <p>Abstract:{paper.paper_info.abstract}</p> 
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;