import React, { useState, useEffect, useRef } from 'react';
import { Document, Page } from 'react-pdf';
import '../pdfjs-config';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

function PDFViewer({ url }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageWidth, setPageWidth] = useState(null);
  const [pageHeight, setPageHeight] = useState(null);
  const [originalDimensions, setOriginalDimensions] = useState(null);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const onPageLoadSuccess = ({ width, height }) => {
    setOriginalDimensions({ width, height });
  };

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      if (!originalDimensions) return;

      const containerWidth = entries[0].contentRect.width;
      let newPageWidth = containerWidth;

       if (containerWidth < 400) {
            newPageWidth *= 0.7;
        } else if (containerWidth < 768) {
            newPageWidth *= 0.8;
        } else if (containerWidth < 1024){
            newPageWidth *= 0.9;
        } else {
            newPageWidth *= 0.95
        }
        newPageWidth -= 34; //remove padding.

      const aspectRatio = originalDimensions.width / originalDimensions.height;
      let newPageHeight = newPageWidth / aspectRatio;


      // Apply zoom level
      setPageWidth(newPageWidth * zoomLevel);
      setPageHeight(newPageHeight * zoomLevel);
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [originalDimensions, zoomLevel]);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setError(null);
  };

  const zoomOut = () => {
    setZoomLevel(prevZoom => Math.max(0.1, prevZoom - 0.1));
  };

  const zoomIn = () => {
    setZoomLevel(prevZoom => prevZoom + 0.1);
  };


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
          flexWrap: 'wrap',
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
            display: 'flex', // Important for centering
            justifyContent: 'center', // Center horizontally
            alignItems: 'center',     // Center vertically

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
        },

      };
    return (
        <div style={styles.container} ref={containerRef}>
            {error ? (
                <div style={{ color: 'red' }}>Error: {error}</div>
            ) : (
                <>
                    {numPages > 0 && (
                        <div style={styles.controls}>
                            <button
                                style={styles.button}
                                disabled={pageNumber <= 1}
                                onClick={() => setPageNumber(p => p - 1)}
                            >
                                Previous
                            </button>
                            <span style={styles.pageInfo}>
                Page {pageNumber} of {numPages}
              </span>
                            <button
                                style={styles.button}
                                disabled={pageNumber >= numPages}
                                onClick={() => setPageNumber(p => p + 1)}
                            >
                                Next
                            </button>
                            <div style={styles.zoomControls}>
                                <button style={styles.button} onClick={zoomOut}>-</button>
                                <span style={styles.zoomText}>
                  {`${Math.round(zoomLevel * 100)}%`}
                </span>
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
                            {/* No extra div needed here */}
                            <Page
                                pageNumber={pageNumber}
                                width={pageWidth}
                                height={pageHeight}
                                renderTextLayer={true}
                                renderAnnotationLayer={true}
                                onLoadSuccess={onPageLoadSuccess}
                            />
                        </Document>
                    </div>
                </>
            )}
        </div>
    );
}

export default PDFViewer;