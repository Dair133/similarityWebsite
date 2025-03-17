import React, { useState, useEffect, useRef } from 'react';
import PDFViewer from './PDFViewer';
import UploadText from './UploadText'
function UploadPDF({ onResultsUpdate, toggleGraphView, pdfFile, onPdfUpload, onClearPdf, results, showGraph }) {
    const [previewUrl, setPreviewUrl] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [dots, setDots] = useState('');
    const fileInputRef = useRef(null);
    const [searchMode, setSearchMode] = useState('upload'); // 'upload' or 'search'
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

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            onPdfUpload(file);  // Update parent state with the File object
            handleUpload(file); // Start the (potentially failing) server request
        } else {
            alert('Please upload a PDF file');
        }
    };

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
                throw new Error(`Upload failed with status: ${response.status}`);
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

    const handleClearPdf = () => {
        onClearPdf();
        setUploadError(null);
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

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

    const styles = {
        container: { width: '75%', backgroundColor: '#53769A', height: '95vh', padding: '2rem', boxSizing: 'border-box' },
        innerBox: { backgroundColor: 'white', borderRadius: '8px', padding: '2rem', width: '90%', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' },
        title: { fontSize: '24px', color: '#333', marginBottom: '1.5rem' },
        input: { width: 0, height: 0, opacity: 0, overflow: 'hidden', position: 'absolute', zIndex: -1 },
        label: { width: '100%', padding: '1rem', border: '2px dashed #ccc', boxSizing: 'border-box', borderRadius: '4px', marginBottom: '1rem', cursor: 'pointer', display: 'inline-block', textAlign: 'center', backgroundColor: '#f9f9f9' },
        fileInfo: { color: '#666', fontSize: '14px', marginBottom: '1rem' },
        viewerArea: { marginTop: '20px', width: '100%' },
        errorText: { color: 'red', marginBottom: '0.5rem', textAlign: 'center' },
        clearButton: {  // Base style for all states of the clear/retry button
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: 'pointer',
            marginBottom: '1rem',
            display: 'inline-block',
            textAlign: 'center',
            width: '100%',
            border: '1px solid #ccc',
            backgroundColor: '#f2f2f2', // Default background (disabled/neutral)
            color: '#999',          // Default text color
        },
        switchContainer: {
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between', // Distribute space
            width: '260px',  // Adjust width as needed
            height: '40px',
            backgroundColor: '#eee',
            borderRadius: '20px', // Pill shape
            padding: '2px',
            marginBottom: '1rem',
            position: 'relative', // For absolute positioning of the slider
        },
        switchOption: {
            flex: '1',  // Each option takes half the space
            textAlign: 'center',
            padding: '8px 16px',
            cursor: 'pointer',
            zIndex: 2, // Ensure text is above the slider
            transition: 'color 0.3s', // Smooth text color transition
            userSelect: 'none',
            color: '#555'
        },
        activeOption: {  // Style for the active text (non selected)
            color: 'white',

        },
        slider: {
            position: 'absolute',
            top: '2px',
            bottom: '2px',
            width: '50%',   // Half the width of the container
            backgroundColor: '#007bff',
            borderRadius: '18px', // Slightly smaller to fit within the container
            transition: 'left 0.3s ease-in-out', // Smooth sliding transition
            zIndex: 1, // Behind the text
        },

    };

    let buttonStyle = { ...styles.clearButton };
    let buttonText = 'Choose a PDF file'; // Default text
    let onClickAction = triggerFileInput;
    let isDisabled = false;

    if (pdfFile) {
        buttonText = 'Clear Current PDF';
        onClickAction = handleClearPdf;
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
                handleClearPdf();
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

    // In the parent component
    const handleSearchResults = (isProcessing, data, errorMessage) => {
        setProcessing(isProcessing);

        if (!isProcessing) {
            if (data) {
                // Handle successful search results
                onResultsUpdate(data);
            } else if (errorMessage) {
                // Handle error
                setUploadError(errorMessage || 'Failed to process the search');
            }
        }
    };

    // Then pass this function to the UploadText component
    <UploadText onSearchSubmit={handleSearchResults} processing={processing} />
    // --- Keyframes for the subtle pulse animation ---
    const keyframes = `@keyframes subtlePulse {
          0% { background-color: #e0e0e0; }      /* Slightly lighter gray */
          50% { background-color: #d0d0d0; }     /* Even lighter gray */
          100% { background-color: #e0e0e0; }    /* Back to slightly lighter */
      }`;

    const handleSearchSubmit = async (query) => {
        try {
            setProcessing(true);
            setUploadError(null);

            // Create form data for the search query
            const formData = new FormData();
            formData.append('query', query);
            formData.append('functionName', 'semanticSearchPapers');

            const response = await fetch('http://localhost:5000/semantic-search', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Search failed with status: ${response.status}`);
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

                        {previewUrl && (
                            <div style={styles.viewerArea}>
                                <h3>PDF Preview:</h3>
                                <PDFViewer url={previewUrl} />
                            </div>
                        )}
                    </>
                ) : (
                    <UploadText onSearchSubmit={handleSearchResults} processing={processing} />
                )}
            </div>
        </div>
    );
}

export default UploadPDF;