import React, { useState } from 'react';
import PDFViewer from './PDFViewer';

function UploadPDF({ onResultsUpdate }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processing, setProcessing] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      // Create a blob URL for the PDF viewer
      const fileUrl = URL.createObjectURL(file);
      setPreviewUrl(fileUrl);
      handleUpload(file);
    } else {
      alert('Please upload a PDF file');
    }
  };

  // Clean up the blob URL when component unmounts or file changes
  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);





  
  const handleUpload = async (file) => {
    try {
      setProcessing(true);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('functionName', 'extractSeedPaperInfo');
      formData.append('pdfPath', file.name);

      const response = await fetch('http://localhost:5000/process-pdf', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      if (onResultsUpdate) {
        onResultsUpdate(data);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to process the file');
    } finally {
      setProcessing(false);
    }
  };


// Define all styles in a single object for clarity and reuse
const styles = {
  container: {
    width: '50%',
    backgroundColor: 'red',
    height:'95vh',
    padding: '2rem',
    boxSizing: 'border-box',
  },
  innerBox: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '2rem',
    width: '90%',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  title: {
    fontSize: '24px',
    color: '#333',
    marginBottom: '1.5rem',
  },
  input: {
    width: '100%',
    padding: '1rem',
    border: '2px dashed #ccc',
    boxSizing: 'border-box',
    borderRadius: '4px',
    marginBottom: '1rem',
    cursor: 'pointer',
  },
  fileInfo: {
    color: '#666',
    fontSize: '14px',
    marginBottom: '1rem',
  },
  processing: {
    color: '#666',
    fontSize: '14px',
    marginBottom: '1rem',
  },
  viewerArea: {
    marginTop: '20px',
    width: '100%',
  },
};


return (
  <div style={styles.container}>
    <div style={styles.innerBox}>
      <h2 style={styles.title}>Upload PDF</h2>
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
      {processing && (
        <div style={styles.processing}>Processing...</div>
      )}
      {previewUrl && (
        <div style={styles.viewerArea}>
          <h3>PDF Preview:</h3>
          <PDFViewer url={previewUrl} />
        </div>
      )}
    </div>
  </div>
);
}

export default UploadPDF;