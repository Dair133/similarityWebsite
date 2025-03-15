import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force';

function NodeGraph({ results, toggleGraphView }) {
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const containerRef = useRef(null);
    const fgRef = useRef(null);

    const styles = {
        container: {
            width: '75%',
            height: '95vh',
            padding: '2rem',
            boxSizing: 'border-box',
            overflow: 'hidden',
        },
        header: {
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem'
        },
        title: {
            fontSize: '24px',
            margin: 0
        },
        button: {
            padding: '8px 16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
        },
        legendContainer: {
            position: 'absolute',
            top: '100px',
            right: '30px',
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            padding: '10px',
            borderRadius: '5px',
            border: '1px solid #ddd',
            zIndex: 1000
        },
        legendItem: {
            display: 'flex',
            alignItems: 'center',
            marginBottom: '5px'
        },
        legendColor: {
            width: '15px',
            height: '15px',
            marginRight: '5px',
            borderRadius: '50%'
        }
    };

    // Update dimensions on mount and window resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                // Account for padding and other elements
                setDimensions({
                    width: width - 40, // Subtract padding from container
                    height: height - 80 // Subtract header and padding
                });
            }
        };

        // Initialize dimensions
        updateDimensions();

        // Add resize listener
        window.addEventListener('resize', updateDimensions);

        // Clean up
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    useEffect(() => {
        if (!results || !results.similarity_results) {
            setGraphData({ nodes: [], links: [] });
            console.warn('No data being passed to NodeGraph component');
            return;
        }

        // Create nodes and links
        const nodes = [];
        const links = [];

        // Add seed paper as central node
        nodes.push({
            id: "seed_paper",
            name: results.title || results.seed_paper?.paper_info?.title || "Seed Paper",
            val: 25, // Larger size for seed paper
            color: "#FF5733", // Distinctive color for seed paper
            isSeed: true
        });

        // Process similarity results and sort by similarity score descending
        const sortedPapers = [...results.test.compared_papers].sort((a, b) => 
            (b.similarity_score || 0) - (a.similarity_score || 0)
        );

        sortedPapers.forEach((paper, index) => {
            // Get a unique ID for the paper
            const paperId = `paper_${index}`;

            // Determine color based on source type
            let color = "#3498DB"; // Default blue color

            if (paper.source_info) {
                const sourceType = paper.source_info.search_type;

                // Color mapping for different search types
                if (sourceType === "core_methodology") {
                    color = "#2ECC71"; // Green for core methodology
                } else if (sourceType === "conceptual_angles") {
                    color = "#9B59B6"; // Purple for conceptual angles
                } else if (sourceType === "poisonPill") {
                    color = "#E74C3C"; // Red for poison pill papers
                }
            }

            // Calculate normalized distance based on rank
            // This ensures a more predictable distribution based on rank
            // Using a steeper curve that accelerates distance after top 10
            let normalizedDistance;
            if (index < 20) {
                // First 10 papers - gentle curve
                normalizedDistance = (index / 20) * 0.3;
            } else {
                // Papers after top 10 - steeper curve
                normalizedDistance = 0.3 + Math.pow((index - 20) / (sortedPapers.length - 20), 0.6) * 0.7;
            }

            // Create node
            nodes.push({
                id: paperId,
                name: paper.paper_info?.title || `Paper ${index + 1}`,
                val: 10, // Smaller size for related papers
                color: color,
                similarity: paper.similarity_score || 0.5, // Default if undefined
                sourceType: paper.source_info?.search_type || "unknown",
                rank: index, // Store the rank based on similarity
                normalizedDistance: normalizedDistance // Store for use in force layout
            });

            // Create link
            links.push({
                source: "seed_paper",
                target: paperId,
                similarity: paper.similarity_score || 0.5, // Default if undefined
                width: Math.max(1, (paper.similarity_score || 0.5) * 5), // Link width based on similarity
                color: color
            });
        });

        setGraphData({ nodes, links });
    }, [results]);

    // Update force simulation to better represent similarity distances
    useEffect(() => {
        if (fgRef.current && graphData.nodes.length > 0) {
            // Use the dynamically calculated dimensions
            const graphWidth = dimensions.width;
            const graphHeight = dimensions.height;
            
            // Configure link force with rank-based distance calculation
            const linkForce = fgRef.current.d3Force('link');
            if (linkForce) {
                linkForce
                    .distance(link => {
                        // For non-seed nodes, use the normalizedDistance
                        if (link.source === "seed_paper" || link.target === "seed_paper") {
                            const node = link.source === "seed_paper" ? 
                                graphData.nodes.find(n => n.id === link.target) : 
                                graphData.nodes.find(n => n.id === link.source);
                            
                            if (node && node.normalizedDistance !== undefined) {
                                // Use a steeper scaling to emphasize differences
                                // Top 10 papers close, then rapid increase
                                return 150 + Math.pow(node.normalizedDistance, 0.8) * 700;
                            }
                        }
                        return 300; // Default fallback
                    })
                    .strength(link => {
                        // Stronger links for more similar nodes
                        if (link.source === "seed_paper" || link.target === "seed_paper") {
                            const node = link.source === "seed_paper" ? 
                                graphData.nodes.find(n => n.id === link.target) : 
                                graphData.nodes.find(n => n.id === link.source);
                            
                            if (node && node.normalizedDistance !== undefined) {
                                // Decrease strength more rapidly as rank increases
                                return Math.max(0.05, 1 - node.normalizedDistance * 1.5);
                            }
                        }
                        return 0.3; // Default fallback
                    });
            }
            
            // Configure charge force for better separation
            const chargeForce = fgRef.current.d3Force('charge');
            if (chargeForce) {
                chargeForce.strength(-800); // Further increased repulsion between nodes
            }
            
            // Use radial force primarily for organizing by rank
            fgRef.current.d3Force('radial', d3.forceRadial()
                .radius(d => {
                    if (d.isSeed) return 0; // Seed paper at center
                    
                    // Use normalizedDistance for predictable radial placement
                    if (d.normalizedDistance !== undefined) {
                        // More dramatic scaling with distinct grouping
                        if (d.rank < 10) {
                            // Top 10 papers - closer together
                            return d.normalizedDistance * Math.min(graphWidth, graphHeight) * 0.7;
                        } else {
                            // Papers after top 10 - spread more aggressively
                            return (0.3 + d.normalizedDistance * 0.7) * Math.min(graphWidth, graphHeight) * 0.9;
                        }
                    }
                    return 300; // Default fallback
                })
                .strength(1.5) // Increased strength for more deterministic layout
                .x(graphWidth / 2)
                .y(graphHeight / 2)
            );
            
            // Add collision force to prevent overlap
            fgRef.current.d3Force('collision', d3.forceCollide()
                .radius(d => Math.sqrt(d.val) * 2 + 10) // Increased collision radius
                .strength(1.0) // Maximum strength
            );
            
            // Reheat the simulation with high alpha for complete reorganization
            fgRef.current.d3ReheatSimulation(1.0);
        }
    }, [graphData, dimensions]); // Add dimensions as dependency

    // Define node rendering
    const nodeCanvasObject = (node, ctx, globalScale) => {
        const nodeR = Math.sqrt(node.val) * 2;

        // Draw node circle
        ctx.beginPath();
        ctx.fillStyle = node.color;
        ctx.arc(node.x, node.y, nodeR, 0, 2 * Math.PI);
        ctx.fill();

        // Draw outline for seed paper
        if (node.isSeed) {
            ctx.beginPath();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.arc(node.x, node.y, nodeR, 0, 2 * Math.PI);
            ctx.stroke();
        }

        // Draw rank number above each non-seed node
        if (!node.isSeed) {
            const rankLabel = `#${node.rank + 1}`; // +1 because rank is zero-based
            ctx.font = '10px Sans-Serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = 'black';
            
            // Add white background for better readability
            const textWidth = ctx.measureText(rankLabel).width;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.fillRect(node.x - textWidth / 2 - 2, node.y - nodeR - 15, textWidth + 4, 14);
            
            // Draw the rank number
            ctx.fillStyle = 'black';
            ctx.fillText(rankLabel, node.x, node.y - nodeR - 5);
        }
        
        // Only show similarity score on hover
        if (!node.isSeed && node === fgRef.current?.hoverNode) {
            const simLabel = node.similarity ? (node.similarity.toFixed(4)) : "";
            ctx.font = '10px Sans-Serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = 'black';
            ctx.fillText(simLabel, node.x, node.y - nodeR - 5);
        }

        // Only draw node label if the node is being hovered over
        if (node === fgRef.current?.hoverNode) {
            const label = node.name;
            const fontSize = node.isSeed ? 14 : 12;

            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Draw background for text
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.fillRect(node.x - textWidth / 2 - 2, node.y + nodeR + 2, textWidth + 4, fontSize + 4);

            // Draw text
            ctx.fillStyle = 'black';
            ctx.fillText(label, node.x, node.y + nodeR + fontSize / 2 + 4);
        }
    };

    // Legend component
    const Legend = () => (
        <div style={styles.legendContainer}>
            <h4 style={{ margin: '0 0 10px 0' }}>Paper Types</h4>
            <div style={styles.legendItem}>
                <div style={{ ...styles.legendColor, backgroundColor: '#FF5733' }}></div>
                <span>Seed Paper</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{ ...styles.legendColor, backgroundColor: '#2ECC71' }}></div>
                <span>Core Methodology</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{ ...styles.legendColor, backgroundColor: '#9B59B6' }}></div>
                <span>Conceptual Angles</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{ ...styles.legendColor, backgroundColor: '#E74C3C' }}></div>
                <span>Poison Pill</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{ ...styles.legendColor, backgroundColor: '#3498DB' }}></div>
                <span>Other</span>
            </div>
        </div>
    );

    return (
        <div ref={containerRef} style={styles.container}>


            {/* <Legend /> */}

            {dimensions.width > 0 && dimensions.height > 0 && (
                <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    nodeRelSize={1}
                    linkWidth={link => link.width}
                    linkColor={link => link.color}
                    nodeCanvasObject={nodeCanvasObject}
                    nodePointerAreaPaint={(node, color, ctx) => {
                        const nodeR = Math.sqrt(node.val) * 2;
                        ctx.fillStyle = color;
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, nodeR + 5, 0, 2 * Math.PI);
                        ctx.fill();
                    }}
                    cooldownTicks={250}  // Increased for better stabilization
                    width={dimensions.width}
                    height={dimensions.height}
                    onResize={() => {
                        // Force graph reheat on resize
                        if (fgRef.current) {
                            fgRef.current.d3ReheatSimulation();
                        }
                    }}
                />
            )}
        </div>
    );
}

export default NodeGraph;