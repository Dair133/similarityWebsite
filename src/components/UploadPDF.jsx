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

  return (
    <div style={{
      width: '40%',
      backgroundColor: 'red',
      minHeight: '100vh',
      padding: '2rem',
      boxSizing: 'border-box',
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '8px',
        padding: '2rem',
        width: '100%',
        maxWidth: '500px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      }}>
        <h2 style={{
          fontSize: '24px',
          color: '#333',
          marginBottom: '1.5rem',
        }}>Upload PDF</h2>
        
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          style={{
            width: '100%',
            padding: '1rem',
            border: '2px dashed #ccc',
            borderRadius: '4px',
            marginBottom: '1rem',
            cursor: 'pointer',
          }}
        />
        
        {selectedFile && (
          <div style={{
            color: '#666',
            fontSize: '14px',
            marginBottom: '1rem',
          }}>
            Selected file: {selectedFile.name}
          </div>
        )}
        
        {processing && (
          <div style={{
            color: '#666',
            fontSize: '14px',
            marginBottom: '1rem',
          }}>
            Processing...
          </div>
        )}

        {previewUrl && <PDFViewer url={previewUrl} />}
      </div>
    </div>
  );
}

export default UploadPDF;