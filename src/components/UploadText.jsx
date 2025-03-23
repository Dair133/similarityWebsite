// UploadText.js (No changes needed here)
import React, { useState, useEffect } from 'react';

function UploadText({ onSearchSubmit, processing }) {
    const [searchQuery, setSearchQuery] = useState('');
    const [dots, setDots] = useState('');  // Now used!

    const handleSubmit = (e) => {
        e.preventDefault();
        if (searchQuery.trim()) {
            onSearchSubmit(searchQuery);
        }
    };

    const handleInputChange = (e) => {
        setSearchQuery(e.target.value);
    };

    // Animated dots (exactly like in UploadPDF)
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


    // Example queries for user reference
    const exampleQueries= [
      "I'm looking for scientific papers that discuss climate change prediction models. I'm particularly interested in models that integrate economic factors, such as the cost of mitigation strategies, the economic impact of climate events, or the influence of economic policies on emissions. I'd prefer papers that use quantitative methods and ideally include case studies or simulations.",
      "I need to find recent research papers (published within the last 5 years) that investigate the relationship between the gut microbiome and mental health. I'm especially interested in studies that use machine learning or other computational methods to analyze microbiome data (like 16S rRNA sequencing or metagenomics) and relate it to mental health outcomes, such as depression, anxiety, or other psychiatric disorders. Ideally, the papers should identify specific microbial biomarkers or predict mental health status.",
      "I'm looking for research papers published in the last 3 years that explore the applications of quantum computing in the field of cryptography. I'm interested in both the use of quantum computers to *break* existing cryptographic systems (quantum cryptanalysis) and the development of new cryptographic methods that are *resistant* to quantum attacks (post-quantum cryptography). Papers discussing quantum key distribution (QKD) or quantum random number generators (QRNGs) would also be relevant. I prefer papers that have a strong theoretical foundation, but experimental implementations or simulations are also of interest."
  ]
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
        boxSizing: 'border-box',
        minHeight: '150px', // Added for larger text area
        resize: 'vertical' // Allows vertical resizing
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
                <textarea // Changed from <input> to <textarea>
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
                        // UploadText.js (rest of the file - no changes, but completing the code)
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