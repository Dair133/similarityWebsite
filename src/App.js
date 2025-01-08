import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);

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

      // const similarityResponse = await fetch('http://localhost:5000/comparePapers', {
      //   method: 'POST',
      // });
     
      

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      //const similarityScore = await similarityResponse.json();
      //console.log(similarityScore)
      console.log(data)
      setResults(data);
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
    }
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
              {/* Display your results here */}
              <pre>{JSON.stringify(results, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;