import React from 'react';
import FadeIn from 'react-fade-in';
import SimplePulseButton from './module/buttons/PulseButton';
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
        interactionButton: { backgroundColor: '#f39c12', color: '#2c3e50', fontWeight: '700', padding: '10px 20px', borderRadius: '50px', textAlign: 'center', display: 'block', margin: '0 auto', cursor: 'pointer', fontFamily: '"Montserrat", sans-serif', fontSize: '16px', border: 'none', textTransform: 'uppercase', letterSpacing: '1px', transition: 'all 0.3s ease', boxShadow: '0 0 0 rgba(243, 156, 18, 0.4)', '&:hover': { animation: 'pulse 1.5s infinite' }, '@keyframes pulse': { '0%': { boxShadow: '0 0 0 0 rgba(243, 156, 18, 0.7)' }, '70%': { boxShadow: '0 0 0 10px rgba(243, 156, 18, 0)' }, '100%': { boxShadow: '0 0 0 0 rgba(243, 156, 18, 0)' } } },
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
        <FadeIn>
            <div style={styles.container}>
                <button style={styles.backButton} onClick={onBack}>← Back to List</button>
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
                                    {paper.comparison_metrics.shared_references.map((reference, i) => (
                                        <span key={i} style={{ ...styles.authorTag, backgroundColor: '#2A5278' }}>
                                            {reference}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </>
                    )}
                    {paper.comparison_metrics.shared_references.length == 0 && (
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
                <div>
                    <h2 style={styles.title}>Actions</h2>
                    <SimplePulseButton onclick={null} buttonText={"Why are these papers similar? 🔎 "} />
                    <br></br>
                    <SimplePulseButton onclick={null} buttonText={"Use this paper as seed paper 🌱 "} />
                    <br></br>
                    <SimplePulseButton onclick={null} buttonText={"Find Paper Online 🌍"} />
                </div>
            </div>
        </FadeIn>
    );
}

export default EnhancedPaperView;
