import React, { useState, useEffect, useRef } from 'react';
import { Document, Page } from 'react-pdf';
import '../pdfjs-config';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

function PDFViewer({ url }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [error, setError] = useState(null);
  const [scale, setScale] = useState(1.5);
  const containerRef = useRef(null);

  // Handle responsive scaling
  useEffect(() => {
    const updateScale = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth;
        // Base scale on container width
        // You can adjust these values based on your needs
        if (containerWidth < 400) {
          setScale(0.8);  // Smaller screens
        } else if (containerWidth < 768) {
          setScale(1.2);  // Medium screens
        } else {
          setScale(1.5);  // Larger screens
        }
      }
    };

    // Initial scale set
    updateScale();

    // Update scale on window resize
    const handleResize = () => {
      updateScale();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const zoomOut = () => setScale(prevScale => Math.max(0.5, prevScale - 0.2));
  const zoomIn = () => setScale(prevScale => Math.min(3, prevScale + 0.2));

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    setError(null);
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
      flexWrap: 'wrap', // Allow controls to wrap on small screens
      justifyContent: 'center',
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
      minWidth: '40px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    documentWrapper: {
      width: '100%',
      maxWidth: '100%',
      border: '1px solid #ddd',
      borderRadius: '4px',
      padding: '1rem',
      backgroundColor: '#fff',
      boxSizing: 'border-box',
      overflow: 'auto',
    },
    zoomControls: {
      display: 'flex',
      gap: '0.5rem',
      alignItems: 'center',
    },
    zoomText: {
      color: '#666',
      minWidth: '60px',
      textAlign: 'center',
    }
  };

  return (
    <div style={styles.container} ref={containerRef}>
      {error ? (
        <div style={{ color: 'red', padding: '1rem' }}>{error}</div>
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
              
              <div style={styles.zoomControls}>
                <button style={styles.button} onClick={zoomOut}>-</button>
                <span style={styles.zoomText}>{Math.round(scale * 100)}%</span>
                <button style={styles.button} onClick={zoomIn}>+</button>
              </div>
            </div>
          )}

          <div style={styles.documentWrapper}>
            <Document
              file={url}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={<div>Loading PDF...</div>}
              error={<div>Failed to load PDF.</div>}
            >
              <Page 
                key={`page_${pageNumber}`}
                pageNumber={pageNumber} 
                scale={scale}
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