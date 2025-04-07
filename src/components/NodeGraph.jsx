import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force'; // Ensure d3-force is imported

// Accept props from ParentDashboard
function NodeGraph({ results, toggleGraphView, onNodeHover, onNodeClick }) {
    // State for graph data, dimensions, and hover information
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    // ---- State to store the currently hovered node's info ----
    const [hoveredNodeInfo, setHoveredNodeInfo] = useState(null); // Stores the node object

    // Refs for container and force graph instance
    const containerRef = useRef(null);
    const fgRef = useRef(null);

    // Component Styles
    const styles = {
        container: {
            width: '100%',
            height: '100%',
            boxSizing: 'border-box',
            overflow: 'hidden',
            backgroundColor: '#0F2741',
            position: 'relative', // Needed for absolute positioning of children
        },
        legendContainer: {
            position: 'absolute',
            top: '20px',
            right: '20px',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: '12px',
            borderRadius: '5px',
            border: '1px solid #ddd',
            zIndex: 1000,
            boxShadow: '0 3px 10px rgba(0, 0, 0, 0.2)',
            fontFamily: '"Source Sans Pro", sans-serif',
        },
        legendHeader: {
            margin: '0 0 10px 0',
            fontFamily: '"Montserrat", sans-serif',
            fontWeight: '600',
            fontSize: '16px',
        },
        legendItem: {
            display: 'flex',
            alignItems: 'center',
            marginBottom: '8px',
            fontSize: '14px',
        },
        legendColor: {
            width: '15px',
            height: '15px',
            marginRight: '8px',
            borderRadius: '50%',
            border: '1px solid rgba(0,0,0,0.1)'
        },
        // ---- Styling for the hover text box ----
        hoverTextBox: {
            position: 'absolute',
            bottom: '20px',
            right: '20px',
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            maxWidth: '250px',
            zIndex: 1010,
            pointerEvents: 'none', // Allows clicks to pass through
            fontFamily: '"Source Sans Pro", sans-serif',
            lineHeight: '1.4',
            opacity: 1,
            transition: 'opacity 0.2s ease-in-out', // Optional fade effect
        },
        hoverTextBoxHidden: { // Style for hiding the text box
             opacity: 0,
        }
    };

    // Effect to update dimensions on mount and resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                setDimensions({ width: width, height: height });
            }
        };
        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Effect to process results data and create graph nodes/links
    useEffect(() => {
        if (!results?.test?.compared_papers) {
            setGraphData({ nodes: [], links: [] });
            console.warn('NodeGraph: results.test.compared_papers not found or invalid.');
            return;
        }

        const nodes = []; const links = [];
        const seedTitle = results.title || results.seed_paper?.paper_info?.title || "Seed Paper";
        nodes.push({ id: "seed_paper", name: seedTitle, val: 25, color: "#FF5733", isSeed: true });

        const papersToCompare = Array.isArray(results.test.compared_papers) ? results.test.compared_papers : [];
        const sortedPapers = [...papersToCompare].sort((a, b) => (b?.similarity_score ?? 0) - (a?.similarity_score ?? 0));

        sortedPapers.forEach((paper, index) => {
            if (!paper || !paper.paper_info) return;
            const paperId = `paper_${index}`;
            let color = "#3498DB";
            const sourceType = paper.source_info?.search_type;
            if (sourceType === "core_methodology") color = "#2ECC71";
            else if (sourceType === "conceptual_angles") color = "#9B59B6";
            else if (sourceType === "poisonPill") color = "#E74C3C";

            // Calculate normalized distance (using your original logic)
             let normalizedDistance;
             const maxIndexForGentleCurve = 20;
             const totalCompared = sortedPapers.length;
             if (totalCompared <= 1) { normalizedDistance = 0; }
             else if (index < maxIndexForGentleCurve) { normalizedDistance = (index / Math.max(1, maxIndexForGentleCurve - 1)) * 0.3; }
             else { const denominator = Math.max(1, totalCompared - maxIndexForGentleCurve); const progress = (index - maxIndexForGentleCurve) / denominator; normalizedDistance = 0.3 + Math.pow(progress, 0.6) * 0.7; }
             normalizedDistance = Math.max(0, Math.min(1, normalizedDistance));

            nodes.push({
                id: paperId,
                name: paper.paper_info.title || `Paper ${index + 1}`,
                val: 10, // Base size value
                color: color,
                similarity: paper.similarity_score ?? 0,
                sourceType: sourceType || "unknown",
                rank: index,
                normalizedDistance: normalizedDistance,
                paperIndex: index, // Index for list linking
                paper: paper       // Full paper data
            });

            links.push({
                source: "seed_paper",
                target: paperId,
                similarity: paper.similarity_score ?? 0,
                width: Math.max(1, (paper.similarity_score ?? 0.5) * 5),
                color: color
            });
        });
        setGraphData({ nodes, links });
    }, [results]);

    // Effect to configure D3 force simulation
     useEffect(() => {
         // Use the CORRECTED structure from the previous response (Response #20)
         // Ensure calculations using 'node' or 'd' are INSIDE the callbacks
         if (fgRef.current && graphData.nodes.length > 1) {
             const graphWidth = dimensions.width;
             const graphHeight = dimensions.height;

             const linkForce = fgRef.current.d3Force('link');
             if (linkForce) {
                 linkForce
                     .distance(link => { /* PASTE YOUR ORIGINAL distance logic using 'node' HERE */
                        const node = link.source === "seed_paper" ? graphData.nodes.find(n => n.id === link.target) : graphData.nodes.find(n => n.id === link.source);
                        if (node && typeof node.normalizedDistance === 'number') { return 150 + Math.pow(node.normalizedDistance, 0.8) * 700; }
                        return 300;
                      })
                     .strength(link => { /* PASTE YOUR ORIGINAL strength logic using 'node' HERE */
                        const node = link.source === "seed_paper" ? graphData.nodes.find(n => n.id === link.target) : graphData.nodes.find(n => n.id === link.source);
                        if (node && typeof node.normalizedDistance === 'number') { return Math.max(0.05, 1 - node.normalizedDistance * 1.5); }
                        return 0.3;
                      });
             }

             const chargeForce = fgRef.current.d3Force('charge');
             if (chargeForce) { chargeForce.strength(-800); /* Your charge */ }

             if (d3.forceRadial) {
                 fgRef.current.d3Force('radial', d3.forceRadial()
                     .radius(d => { /* PASTE YOUR ORIGINAL radius logic using 'd' HERE */
                         if (d.isSeed) return 0; if (typeof d.normalizedDistance === 'number') { const minDim = Math.min(graphWidth, graphHeight); if (d.rank < 10) { return d.normalizedDistance * minDim * 0.7; } else { return (0.3 + d.normalizedDistance * 0.7) * minDim * 0.9; } } return 300;
                      })
                     .strength(1.5).x(graphWidth / 2).y(graphHeight / 2)); /* Your strength/center */
             }

             if (d3.forceCollide) {
                 fgRef.current.d3Force('collision', d3.forceCollide()
                     .radius(d => { /* PASTE YOUR ORIGINAL collision radius logic using 'd' HERE */ return Math.sqrt(d.val) * 2 + 10; })
                     .strength(1.0)); /* Your strength */
             }

             fgRef.current.d3ReheatSimulation(1.0); // Your alpha
         }
     }, [graphData, dimensions]);

    // Function to define how nodes are drawn
    const nodeCanvasObject = (node, ctx, globalScale) => {
        const baseNodeR = Math.sqrt(node.val) * 2;
        const isHovered = fgRef.current?.hoverNode === node;
        // Apply subtle scaling on hover (optional)
        let drawRadius = isHovered && !node.isSeed ? baseNodeR * 1.10 : baseNodeR;

        // Draw main circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, drawRadius, 0, 2 * Math.PI, false);
        ctx.fillStyle = node.color || 'grey';
        ctx.fill();

        // Draw seed outline (using base radius)
        if (node.isSeed) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5 / globalScale;
            ctx.beginPath();
            ctx.arc(node.x, node.y, baseNodeR, 0, 2 * Math.PI, false);
            ctx.stroke();
        }

        // Draw Rank Label (using drawRadius for position)
        const labelThreshold = 5;
        if (!node.isSeed && (globalScale > labelThreshold || graphData.nodes.length < 50)) {
            const rankLabel = `#${node.rank + 1}`;
            const fontSize = Math.max(6, 9 / globalScale);
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            const labelY = node.y - drawRadius - (fontSize / 2 + 2 / globalScale);
            const textWidth = ctx.measureText(rankLabel).width;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.fillRect(node.x - textWidth / 2 - 1 / globalScale, labelY - fontSize / 2 - 1 / globalScale, textWidth + 2 / globalScale, fontSize + 2 / globalScale);
            ctx.fillStyle = 'black'; ctx.fillText(rankLabel, node.x, labelY);
        }

        // Draw Hover Label (using drawRadius for position)
        if (isHovered) {
            const label = node.name || '';
            const fontSize = Math.max(8, 12 / globalScale);
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            const labelY = node.y + drawRadius + fontSize * 0.8;
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'; // Background
            ctx.fillRect(node.x - textWidth / 2 - 2, labelY - fontSize/2 - 2 , textWidth + 4, fontSize + 4);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'; // Text
            ctx.fillText(label, node.x, labelY);
        }
    };

    // Handler for node hover events
    const handleNodeHover = (node) => {
        // Update local state for the hover text box display
        setHoveredNodeInfo(node); // Store node object or null

        // Call parent handler for list scrolling
        if (typeof onNodeHover === 'function') {
            onNodeHover(node ? node.paperIndex : null);
        }
        // Update cursor
        if (containerRef.current) {
            containerRef.current.style.cursor = node ? 'pointer' : 'grab';
        }
    };

    // Handler for node click events
    const handleNodeClick = (node) => {
        // Call parent handler for enhanced view
       if (node && !node.isSeed && typeof onNodeClick === 'function') {
            onNodeClick(node.paper); // Pass paper object
       }
    };

    // Legend component definition
    const Legend = () => (
         <div style={styles.legendContainer}>
             <h4 style={styles.legendHeader}>Paper Types</h4>
              {[ /* Your legend items */
                 { color: '#FF5733', label: 'Seed Paper' },
                 { color: '#2ECC71', label: 'Core Methodology' },
                 { color: '#9B59B6', label: 'Conceptual Angles' },
                 { color: '#E74C3C', label: 'Poison Pill' },
                 { color: '#3498DB', label: 'Other' },
              ].map(item => (
                 <div key={item.label} style={styles.legendItem}>
                     <div style={{ ...styles.legendColor, backgroundColor: item.color }}></div>
                     <span>{item.label}</span>
                 </div>
              ))}
         </div>
     );

    // Render the component
    return (
        <div ref={containerRef} style={styles.container}>
            {graphData.nodes.length > 1 && <Legend />}

            {dimensions.width > 0 && dimensions.height > 0 && (
                <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    width={dimensions.width}
                    height={dimensions.height}
                    backgroundColor="white"
                    nodeRelSize={1}
                    nodeCanvasObject={nodeCanvasObject}
                    linkWidth={link => link.width}
                    linkColor={link => link.color || '#ffffff44'}
                    nodePointerAreaPaint={(node, color, ctx) => {
                         const nodeR = Math.sqrt(node.val) * 2;
                         ctx.fillStyle = color; ctx.beginPath();
                         ctx.arc(node.x, node.y, nodeR + 6, 0, 2 * Math.PI, false);
                         ctx.fill();
                     }}
                    onNodeHover={handleNodeHover} // Use updated handler
                    onNodeClick={handleNodeClick} // Use click handler
                    cooldownTicks={250}
                    enableNodeDrag={false} // Keep dragging disabled
                    enablePointerInteraction={true}
                    enableZoomPanInteraction={true}
                />
            )}

            {/* Hover Text Box - Rendered conditionally */}
            <div style={{
                 ...styles.hoverTextBox,
                 // Apply hidden style if no valid node is hovered
                 ...( (!hoveredNodeInfo || hoveredNodeInfo.isSeed) && styles.hoverTextBoxHidden )
             }}>
                 {/* Only render text if a valid (non-seed) node is hovered */}
                {hoveredNodeInfo && !hoveredNodeInfo.isSeed && (
                    <>
                        Click For Enhanced Paper view for: <br />
                        <strong>{hoveredNodeInfo.name}</strong> {/* Display hovered node name */}
                    </>
                )}
            </div>

        </div> // End container div
    );
}

export default NodeGraph;