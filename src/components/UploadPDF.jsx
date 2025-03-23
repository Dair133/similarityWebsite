// UploadPDF.js
import React, { useState, useEffect, useRef } from 'react';
import PDFViewer from './PDFViewer';
import UploadText from './UploadText';

function UploadPDF({ onResultsUpdate, toggleGraphView, pdfFile, onPdfUpload, onClearPdf, results, showGraph }) {
    const [previewUrl, setPreviewUrl] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [dots, setDots] = useState('');
    const fileInputRef = useRef(null);
    const [searchMode, setSearchMode] = useState('upload'); // 'upload' or 'search'

    // Generate preview URL
    useEffect(() => {
        let fileUrl;
        if (pdfFile) {
            fileUrl = URL.createObjectURL(pdfFile);
            setPreviewUrl(fileUrl);
        } else {
            setPreviewUrl(null);
        }
        return () => fileUrl && URL.revokeObjectURL(fileUrl);
    }, [pdfFile]);

    // Handle file change
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            onPdfUpload(file);
            handleUpload(file);
        } else {
            alert('Please upload a PDF file');
        }
    };

    // Handle PDF upload
    const handleUpload = async (file) => {
        try {
            setProcessing(true);
            setUploadError(null);
            const formData = new FormData();
            formData.append('file', file);
            formData.append('functionName', 'extractSeedPaperInfo');
            formData.append('pdfPath', file.name);

            const response = await fetch('http://localhost:5000/process-pdf', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const data = await response.json();
            onResultsUpdate(data);
        } catch (error) {
            console.error('Error:', error);
            setUploadError(error.message || 'Failed to process the file');
        } finally {
            setProcessing(false);
        }
    };

    // Trigger file input
    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    // Animated dots
    useEffect(() => {
        let intervalId;
        if (processing) {
            intervalId = setInterval(() => {
                setDots(prevDots => (prevDots === '...' ? '' : prevDots + '.'));
            }, 500);
        } else {
            setDots('');
        }
        return () => clearInterval(intervalId);
    }, [processing]);

    // Perform semantic search (CORRECTED ENDPOINT)
    const handleSearchSubmit = async (query) => {
        try {
            setProcessing(true);
            setUploadError(null);

            const requestData = {
                query: query,
                functionName: 'semanticSearchPapers', // Include functionName
            };

            const response = await fetch('http://localhost:5000/natural-language-search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Set to application/json
                },
                body: JSON.stringify(requestData), // Stringify the object
            });

            if (!response.ok) {
                throw new Error(`Search failed: ${response.status}`);
            }

            const data = await response.json();
            onResultsUpdate(data);
        } catch (error) {
            console.error('Error:', error);
            setUploadError(error.message || 'Failed to process the search');
        } finally {
            setProcessing(false);
        }
    };

   // Styles (No changes needed here)
    const styles = {
        container: { 
          width: '100%', 
          height: '100%',
          backgroundColor: '#53769A', 
          boxSizing: 'border-box',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1rem'
        },
        innerBox: { 
          backgroundColor: 'white', 
          borderRadius: '8px', 
          padding: '1.5rem', 
          width: '90%',
          height: '90%',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',  // Hide overflow at this level
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' 
        },
        title: { 
          fontSize: '22px', 
          color: '#333', 
          marginBottom: '1rem',
          fontWeight: 'bold'
        },
        input: { 
          width: 0, 
          height: 0, 
          opacity: 0, 
          overflow: 'hidden', 
          position: 'absolute', 
          zIndex: -1 
        },
        label: { 
          width: '100%', 
          padding: '1rem', 
          border: '2px dashed #ccc', 
          boxSizing: 'border-box', 
          borderRadius: '4px', 
          marginBottom: '1rem', 
          cursor: 'pointer', 
          display: 'inline-block', 
          textAlign: 'center', 
          backgroundColor: '#f9f9f9',
          fontSize: '16px'
        },
        fileInfo: { 
          color: '#666', 
          fontSize: '14px', 
          marginBottom: '1rem' 
        },
        contentArea: {
          flex: '1 1 auto',  // Allow this to grow and shrink as needed
          overflow: 'auto',  // Add scrollbars only to content
          marginTop: '10px'
        },
        viewerArea: { 
          width: '100%',
          height: 'calc(100% - 30px)', // Subtract the height of the heading
          overflow: 'hidden'
        },
        errorText: { 
          color: 'red', 
          marginBottom: '0.5rem', 
          textAlign: 'center' 
        },
        clearButton: {  // Base style for all states of the clear/retry button
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: 'pointer',
          marginBottom: '1rem',
          display: 'inline-block',
          textAlign: 'center',
          width: '100%',
          border: '1px solid #ccc',
          backgroundColor: '#f2f2f2', // Default background
          color: '#999',          // Default text color
          fontSize: '16px'
        },
        switchContainer: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '260px',
          height: '40px',
          backgroundColor: '#eee',
          borderRadius: '20px',
          padding: '2px',
          marginBottom: '1rem',
          position: 'relative',
        },
        switchOption: {
          flex: '1',
          textAlign: 'center',
          padding: '8px 16px',
          cursor: 'pointer',
          zIndex: 2,
          transition: 'color 0.3s',
          userSelect: 'none',
          color: '#555',
          fontSize: '14px',
          fontWeight: '500'
        },
        activeOption: {
          color: 'white',
        },
        slider: {
          position: 'absolute',
          top: '2px',
          bottom: '2px',
          width: '50%',
          backgroundColor: '#007bff',
          borderRadius: '18px',
          transition: 'left 0.3s ease-in-out',
          zIndex: 1,
        }
      };
    
      // Dynamic button styling
        let buttonStyle = { ...styles.clearButton };
        let buttonText = 'Choose a PDF file'; // Default text
        let onClickAction = triggerFileInput;
        let isDisabled = false;
    
        if (pdfFile) {
            buttonText = 'Clear Current PDF';
            onClickAction = onClearPdf;
            buttonStyle.backgroundColor = '#ffdddd';
            buttonStyle.color = '#333';
            buttonStyle.border = '1px solid #ffaaaa';
            isDisabled = false;
    
            if (processing) {
                buttonText = `Processing${dots}`;
                // Subtle animation during processing
                buttonStyle.animation = 'subtlePulse 1.5s infinite';
                isDisabled = true;
            } else if (uploadError) {
                buttonText = 'Clear PDF & Retry';
                onClickAction = () => {  // Correctly resets for retry
                    onClearPdf();
                    triggerFileInput();
                }
                buttonStyle.backgroundColor = '#ffdddd'; // Keep consistent with Clear
                buttonStyle.color = '#333';
                buttonStyle.border = '1px solid #ffaaaa';
                isDisabled = false;
            } else if (results !== null) { //Success
                buttonStyle.backgroundColor = '#ffdddd';
                buttonStyle.color = '#333';
                buttonStyle.border = '1px solid #ffaaaa';
                isDisabled = false;
            }
        }
    
        // Keyframes for the subtle pulse animation
        const keyframes = `@keyframes subtlePulse {
                0% { background-color: #e0e0e0; }      /* Slightly lighter gray */
                50% { background-color: #d0d0d0; }     /* Even lighter gray */
                100% { background-color: #e0e0e0; }    /* Back to slightly lighter */
            }`;

    return (
        <div style={styles.container}>
            <style>{keyframes}</style>
            <div style={styles.innerBox}>
                {/* Mode toggle */}
                <div style={styles.switchContainer}>
                    <div
                        style={{
                            ...styles.switchOption,
                            ...(searchMode === 'upload' ? styles.activeOption : {})
                        }}
                        onClick={() => setSearchMode('upload')}
                    >
                        Upload PDF
                    </div>
                    <div
                        style={{
                            ...styles.switchOption,
                            ...(searchMode === 'search' ? styles.activeOption : {})
                        }}
                        onClick={() => setSearchMode('search')}
                    >
                        Semantic Search
                    </div>
                    <div
                        style={{
                            ...styles.slider,
                            left: searchMode === 'upload' ? '2px' : '50%'
                        }}
                    />
                </div>

                {/* Render either PDF upload UI or Semantic search UI based on mode */}
                {searchMode === 'upload' ? (
                    <>
                        <h2 style={styles.title}>Upload PDF</h2>

                        {!pdfFile && (
                            <label htmlFor="file-upload" style={styles.label}>
                                {buttonText}
                                <input
                                    type="file"
                                    id="file-upload"
                                    accept=".pdf"
                                    onChange={handleFileChange}
                                    style={styles.input}
                                    ref={fileInputRef}
                                />
                            </label>
                        )}

                        {pdfFile && (
                            <button
                                style={buttonStyle}
                                onClick={onClickAction}
                                disabled={isDisabled}
                            >
                                {buttonText}
                            </button>
                        )}

                        {uploadError && (<div style={styles.errorText}>{uploadError}</div>)}

                        {pdfFile && (<div style={styles.fileInfo}>Selected file: {pdfFile.name}</div>)}

                        <div style={styles.contentArea}>
                            {previewUrl && (
                                <div style={styles.viewerArea}>
                                    <h3 style={{margin: '0 0 10px 0', fontSize: '16px'}}>PDF Preview:</h3>
                                    <PDFViewer url={previewUrl} />
                                </div>
                            )}
                        </div>
                    </>
                ) : (
                    <div style={styles.contentArea}>
                        {/* Pass processing and onSearchSubmit directly */}
                        <UploadText onSearchSubmit={handleSearchSubmit} processing={processing} />
                    </div>
                )}
            </div>
        </div>
    );
}

export default UploadPDF;