import React from 'react';

function EnhancedPaperView({ paper, onBack }) {
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
        },
        authors: {
            fontSize: '16px',
            marginBottom: '8px',
            color: '#81A4CD',
        },
        abstract: {
            fontSize: '14px',
            lineHeight: '1.6',
            marginBottom: '15px',
        },
        section: {
            marginBottom: '15px',
        },
        sectionTitle: {
            fontSize: '18px',
            fontWeight: 'bold',
            marginBottom: '8px',
            color: '#EEE8D9'
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
    };

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

    return (
        <div style={styles.container}>
            <button style={styles.backButton} onClick={onBack}>Back to List</button>
            <h2 style={styles.title}>{paper.paper_info.title}</h2>
            <p style={styles.authors}>
                <strong>Authors:</strong> {formatAuthors(paper.paper_info.authors)}
            </p>
            <p style={styles.abstract}>
                <strong>Abstract:</strong> {paper.paper_info.abstract}
            </p>
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
                                {paper.comparison_metrics.shared_references.slice(0, 5).map((reference, i) => (
                                    <span key={i} style={{ ...styles.authorTag, backgroundColor: '#2A5278' }}>
                                        {reference}
                                    </span>
                                ))}
                                {paper.comparison_metrics.shared_references.length > 5 && (
                                    <span style={{ color: '#81A4CD', fontSize: '13px', marginLeft: '5px', alignSelf: 'center' }}>
                                        ... {paper.comparison_metrics.shared_references.length - 5} more
                                    </span>
                                )}
                            </div>
                        </div>
                    </>
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
        </div>
    );
}

export default EnhancedPaperView;
