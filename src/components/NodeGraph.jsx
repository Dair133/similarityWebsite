import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force';

function NodeGraph({ results, toggleGraphView }) {
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const fgRef = useRef(null);

    const styles = {
        container: {
            width: '75%',
            backgroundColor: '#f5f5f5',
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

        // Process similarity results
        results.similarity_results.forEach((paper, index) => {
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
            
            // Create node
            nodes.push({
                id: paperId,
                name: paper.paper_info.title || `Paper ${index + 1}`,
                val: 10, // Smaller size for related papers
                color: color,
                similarity: paper.similarity_score || 0.5, // Default if undefined
                sourceType: paper.source_info?.search_type || "unknown"
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

    // Customize force simulation
    useEffect(() => {
        if (fgRef.current && graphData.nodes.length > 0) {
            // Get the link force and modify it
            const linkForce = fgRef.current.d3Force('link');
            if (linkForce) {
                linkForce
                    .distance(link => 200 * (1 - Math.pow(link.similarity, 2))) // Non-linear distance based on similarity
                    .strength(link => 0.1 + link.similarity * 0.5); // Stronger links for higher similarity
            }
            
            // Modify charge force for repulsion
            const chargeForce = fgRef.current.d3Force('charge');
            if (chargeForce) {
                chargeForce.strength(-120);
            }
            
            // Modify center force
            const centerForce = fgRef.current.d3Force('center');
            if (centerForce) {
                centerForce.strength(0.15);
            }
            
            // Get graph dimensions from the container
            const graphWidth = fgRef.current.graphWidth || window.innerWidth * 0.5 - 80;
            const graphHeight = fgRef.current.graphHeight || window.innerHeight * 0.95 - 120;
            
            // Add or update radial force
            fgRef.current.d3Force('radial', d3.forceRadial()
                .radius(d => d.isSeed ? 0 : 200 * (1 - Math.pow(d.similarity || 0.5, 2)))
                .strength(d => d.isSeed ? 0 : 0.8)
                .x(graphWidth / 2)
                .y(graphHeight / 2)
            );
            
            // Reheat the simulation
            fgRef.current.d3ReheatSimulation();
        }
    }, [graphData]);

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
            ctx.fillRect(node.x - textWidth/2 - 2, node.y + nodeR + 2, textWidth + 4, fontSize + 4);
            
            // Draw text
            ctx.fillStyle = 'black';
            ctx.fillText(label, node.x, node.y + nodeR + fontSize/2 + 4);
        }
    };

    // Legend component
    const Legend = () => (
        <div style={styles.legendContainer}>
            <h4 style={{margin: '0 0 10px 0'}}>Paper Types</h4>
            <div style={styles.legendItem}>
                <div style={{...styles.legendColor, backgroundColor: '#FF5733'}}></div>
                <span>Seed Paper</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{...styles.legendColor, backgroundColor: '#2ECC71'}}></div>
                <span>Core Methodology</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{...styles.legendColor, backgroundColor: '#9B59B6'}}></div>
                <span>Conceptual Angles</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{...styles.legendColor, backgroundColor: '#E74C3C'}}></div>
                <span>Poison Pill</span>
            </div>
            <div style={styles.legendItem}>
                <div style={{...styles.legendColor, backgroundColor: '#3498DB'}}></div>
                <span>Other</span>
            </div>
        </div>
    );

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2 style={styles.title}>Paper Similarity Network</h2>
                <button style={styles.button} onClick={toggleGraphView}>
                    Switch to Seed Paper View
                </button>
            </div>
            
            <Legend />
            
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
                cooldownTicks={100}
                width={window.innerWidth * 0.5 - 80} // Adjust for container width and padding
                height={window.innerHeight * 0.95 - 120} // Adjust for header and container padding
                // onEngineStop={() => {
                //     if (fgRef.current) {
                //         setTimeout(() => fgRef.current.zoomToFit(400, 100), 500);
                //     }
                // }}
            />
        </div>
    );
}

export default NodeGraph;