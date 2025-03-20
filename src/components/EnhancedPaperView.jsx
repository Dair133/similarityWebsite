import React, { useState } from 'react';
import FadeIn from 'react-fade-in';
import SimplePulseButton from './module/buttons/PulseButton';

function EnhancedPaperView({ paper, onBack, seedPaper }) {
    // Add state for similarity explanation
    const [similarityExplanation, setSimilarityExplanation] = useState(null);
    const [isExplaining, setIsExplaining] = useState(false);
    const [explanationError, setExplanationError] = useState(null);

    const styles = {
        container: {
            backgroundColor: '#14304D',
            padding: '15px',
            borderRadius: '6px',
            color: '#F7F3E9',
            width: '100%',
            boxSizing: 'border-box',
        },
        title: {
            fontSize: '24px',
            fontWeight: 'bold',
            marginBottom: '10px',
            color: '#EEE8D9',
            fontFamily: '"Montserrat", sans-serif',
        },
        authors: {
            fontSize: '16px',
            marginBottom: '8px',
            color: '#81A4CD',
            fontFamily: '"Source Sans Pro", sans-serif',
        },
        abstract: {
            fontSize: '14px',
            lineHeight: '1.6',
            marginBottom: '15px',
            fontFamily: '"Source Sans Pro", sans-serif',
        },
        section: {
            marginBottom: '15px',
        },
        sectionTitle: {
            fontSize: '18px',
            fontWeight: 'bold',
            marginBottom: '8px',
            color: '#EEE8D9',
            fontFamily: '"Montserrat", sans-serif',
        },
        metric: {
            color: '#81A4CD',
            fontWeight: 'bold',
        },
        backButton: {
            backgroundColor: '#3E7CB9',
            border: 'none',
            color: '#F7F3E9',
            padding: '10px 20px',
            textAlign: 'center',
            textDecoration: 'none',
            display: 'inline-block',
            fontSize: '16px',
            margin: '4px 2px',
            cursor: 'pointer',
            borderRadius: '4px',
            fontFamily: '"Montserrat", sans-serif',
            fontWeight: '500',
        },
        overlapItem: {
            marginBottom: '8px',
        },
        sectionLabel: {
            fontSize: '13px',
            color: '#81A4CD',
            fontFamily: '"Montserrat", sans-serif',
            fontWeight: '500',
            marginBottom: '8px',
        },
        authorTag: {
            display: 'inline-block',
            padding: '5px 10px',
            backgroundColor: '#38729E',
            color: '#F7F3E9',
            borderRadius: '4px',
            fontSize: '13px',
            fontFamily: '"Source Sans Pro", sans-serif',
            margin: '2px 4px 2px 0',
        },
        explanationContainer: {
            backgroundColor: '#1A3A5F',
            borderRadius: '6px',
            padding: '15px',
            marginTop: '15px',
            marginBottom: '15px',
            border: '1px solid #3E7CB9',
            fontFamily: '"Source Sans Pro", sans-serif',
        },
        explanationTitle: {
            fontSize: '18px',
            fontWeight: 'bold',
            marginBottom: '10px',
            color: '#EEE8D9',
            fontFamily: '"Montserrat", sans-serif',
            display: 'flex',
            alignItems: 'center',
        },
        loadingSpinner: {
            display: 'inline-block',
            width: '20px',
            height: '20px',
            marginLeft: '10px',
            border: '3px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '50%',
            borderTop: '3px solid #EEE8D9',
            animation: 'spin 1s linear infinite',
        },
        explanationText: {
            lineHeight: '1.6',
            whiteSpace: 'pre-wrap',
        },
        explanationError: {
            color: '#ff6b6b',
            fontStyle: 'italic',
            marginTop: '5px',
        },
        keyPoint: {
            backgroundColor: 'rgba(62, 124, 185, 0.3)',
            padding: '10px',
            borderRadius: '4px',
            marginTop: '5px',
            marginBottom: '10px',
            borderLeft: '3px solid #3E7CB9',
        },
        infoTag: {
            display: 'inline-block',
            backgroundColor: '#2A5278',
            color: '#F7F3E9',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '12px',
            marginRight: '5px',
        },
        sectionDivider: {
            height: '1px',
            width: '100%',
            backgroundColor: '#3E7CB9',
            margin: '20px 0',
            opacity: 0.5,
        },
    };

    // Add keyframes for spinner animation
    const keyframes = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;

    // Helper function to safely format authors
    const formatAuthors = (authors) => {
        if (!authors) return '';
        if (typeof authors === 'string') return authors;
        if (Array.isArray(authors)) return authors.join(', ');
        return JSON.stringify(authors);
    };

    // Helper function to format source type text
    const formatSourceType = (sourceType) => {
        if (!sourceType) return '';
        return sourceType
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    // Function to explain similarity
    const explainSimilarity = async () => {
        // Reset states
        setIsExplaining(true);
        setSimilarityExplanation(null);
        setExplanationError(null);

        try {
            // Get seed paper info - make sure we have the necessary properties
            const seedPaperInfo = seedPaper?.paper_info || {};
            const seedPaperTitle = seedPaperInfo.title || "Unknown Seed Paper";
            const seedPaperAbstract = seedPaperInfo.abstract || "";

            // Get current paper info
            const currentPaperTitle = paper.paper_info.title || "Unknown Paper";
            const currentPaperAbstract = paper.paper_info.abstract || "";

            // Create request data
            const requestData = {
                seed_paper: {
                    title: seedPaperTitle,
                    abstract: seedPaperAbstract
                },
                current_paper: {
                    title: currentPaperTitle,
                    abstract: currentPaperAbstract
                },
                similarity_metrics: {
                    similarity_score: paper.similarity_score,
                    shared_references: paper.comparison_metrics.shared_reference_count,
                    shared_authors: paper.comparison_metrics.shared_authors || []
                }
            };

            // Make API call
            const response = await fetch('http://localhost:5000/explain-similarity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });

            if (!response.ok) {
                throw new Error(`Request failed with status: ${response.status}`);
            }

            const data = await response.json();
            setSimilarityExplanation(data.explanation);
        } catch (error) {
            console.error('Error explaining similarity:', error);
            setExplanationError(error.message || 'Failed to generate similarity explanation');
        } finally {
            setIsExplaining(false);
        }
    };


    const findPaperOnleine = async () => {


    }



    return (
        <FadeIn>
            <style>{keyframes}</style>
            <div style={styles.container}>
                <button style={styles.backButton} onClick={onBack}>← Back to List</button>
                <h2 style={styles.title}>{paper.paper_info.title}</h2>
                <p style={styles.authors}>
                    <strong>Authors:</strong> {formatAuthors(paper.paper_info.authors)}
                </p>
                <p style={styles.abstract}>
                    <strong>Abstract:</strong> {paper.paper_info.abstract}
                </p>

                {/* Similarity Explanation Section */}
                {(similarityExplanation || isExplaining) && (
                    <div style={styles.explanationContainer}>
                        <h3 style={styles.explanationTitle}>
                            Why are these papers similar?
                            {isExplaining && <span style={styles.loadingSpinner}></span>}
                        </h3>

                        {explanationError && (
                            <p style={styles.explanationError}>{explanationError}</p>
                        )}

                        {similarityExplanation && (
                            <div style={styles.explanationText}>
                                {similarityExplanation}

                            </div>
                        )}

                    </div>
                )}

                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Similarity Metrics</h3>
                    <p>
                        <strong>Similarity Score:</strong> <span style={styles.metric}>{paper.similarity_score}</span>
                    </p>
                    <p>
                        <strong>Shared References:</strong> <span style={styles.metric}>{paper.comparison_metrics.shared_reference_count}</span>
                    </p>
                    <p>
                        <strong>Shared Citations:</strong> <span style={styles.metric}>{paper.comparison_metrics.shared_citation_count}</span>
                    </p>
                    <p>
                        <strong>Shared Authors:</strong> <span style={styles.metric}>{paper.comparison_metrics.shared_author_count}</span>
                    </p>
                    <div style={styles.overlapItem}>
                        {paper.comparison_metrics.shared_authors && paper.comparison_metrics.shared_authors.length > 0 && (
                            <>
                                <div style={styles.sectionLabel}>Shared Authors:</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                                    {paper.comparison_metrics.shared_authors.map((author, i) => (
                                        <span key={i} style={styles.authorTag}>{author}</span>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                    {paper.comparison_metrics.shared_references && paper.comparison_metrics.shared_references.length > 0 && (
                        <>
                            <div style={styles.sectionLabel}>Shared References:</div>
                            <div style={styles.overlapItem}>
                                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                                    {paper.comparison_metrics.shared_references.map((reference, i) => (
                                        <span key={i} style={{ ...styles.authorTag, backgroundColor: '#2A5278' }}>
                                            {reference}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </>
                    )}
                    {paper.comparison_metrics.shared_references && paper.comparison_metrics.shared_references.length == 0 && (
                        <div style={styles.sectionLabel}>No shared references</div>
                    )}
                </div>

                {paper.source_info && (
                    <div style={styles.section}>
                        <h3 style={styles.sectionTitle}>Source Information</h3>
                        <p>
                            <strong>Search Term:</strong> {Array.isArray(paper.source_info.search_term) ? paper.source_info.search_term.join(', ') : paper.source_info.search_term}
                        </p>
                        <p><strong>Search Type:</strong> {formatSourceType(paper.source_info.search_type)}</p>
                    </div>
                )}

                <div style={styles.sectionDivider}></div>

                <div>
                    <h2 style={styles.title}>Actions</h2>
                    <SimplePulseButton
                        onClick={explainSimilarity}
                        buttonText={isExplaining ? "Generating explanation..." : "Why are these papers similar? 🔎"}
                        customStyle={{
                            fontSize: '14px',
                            fontWeight: '700',
                            backgroundColor: '#94B4DC',
                            width: '100%',
                            marginBottom: '15px',
                            opacity: isExplaining ? 0.7 : 1,
                            cursor: isExplaining ? 'default' : 'pointer'
                        }}
                        disabled={isExplaining}
                    />

                    <SimplePulseButton
                        onClick={null}
                        buttonText={"Use this paper as seed paper 🌱"}
                        customStyle={{
                            fontSize: '14px',
                            fontWeight: '700',
                            backgroundColor: '#94B4DC',
                            width: '100%',
                            marginBottom: '15px'
                        }}
                    />

                    <SimplePulseButton
                        onClick={null}
                        buttonText={"Find Paper Online 🌍"}
                        customStyle={{
                            fontSize: '14px',
                            fontWeight: '700',
                            backgroundColor: '#94B4DC',
                            width: '100%'
                        }}
                    />
                </div>
            </div>
        </FadeIn>
    );
}

export default EnhancedPaperView;