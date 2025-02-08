import React, { useState } from 'react';
import { Document, Page } from 'react-pdf';
import '../pdfjs-config';  // Import the config

function PDFViewer({ url }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [error, setError] = useState(null);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    setError(null);
  }

  function onDocumentLoadError(error) {
    console.error('PDF Load Error:', error);
    setError('Failed to load PDF. Please try again.');
  }

  const styles = {
    container: {
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '1rem',
    },
    controls: {
      display: 'flex',
      gap: '1rem',
      alignItems: 'center',
      marginBottom: '1rem',
    },
    pageInfo: {
      color: '#666',
    },
    button: {
      padding: '0.5rem 1rem',
      backgroundColor: '#f0f0f0',
      border: '1px solid #ddd',
      borderRadius: '4px',
      cursor: 'pointer',
    },
    documentWrapper: {
      maxWidth: '100%',
      border: '1px solid #ddd',
      borderRadius: '4px',
      padding: '1rem',
      backgroundColor: '#fff',
    },
    errorMessage: {
      color: 'red',
      textAlign: 'center',
      padding: '1rem',
    }
  };

  return (
    <div style={styles.container}>
      {error ? (
        <div style={styles.errorMessage}>{error}</div>
      ) : (
        <>
          {numPages && (
            <div style={styles.controls}>
              <button
                style={styles.button}
                disabled={pageNumber <= 1}
                onClick={() => setPageNumber(prev => prev - 1)}
              >
                Previous
              </button>
              <span style={styles.pageInfo}>
                Page {pageNumber} of {numPages}
              </span>
              <button
                style={styles.button}
                disabled={pageNumber >= numPages}
                onClick={() => setPageNumber(prev => prev + 1)}
              >
                Next
              </button>
            </div>
          )}

          <div style={styles.documentWrapper}>
            <Document
              file={url}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div>Loading PDF...</div>}
              error={<div>Failed to load PDF.</div>}
            >
              <Page 
                key={`page_${pageNumber}`}
                pageNumber={pageNumber} 
                width={500}
                renderTextLayer={true}
                renderAnnotationLayer={true}
              />
            </Document>
          </div>
        </>
      )}
    </div>
  );
}

export default PDFViewer;