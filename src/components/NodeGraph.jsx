import React, { useEffect, useState, useRef } from 'react'; // Import useRef
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force';

function NodeGraph({ results, toggleGraphView }) {
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const fgRef = useRef(null); // Create a ref for the ForceGraph2D instance

    const styles = {
        container: {
            width: '50%', // Consider making this responsive
            backgroundColor: 'white', // Better for 2D
            height: '95vh',  // Use vh units for viewport height
            padding: '2rem',
            boxSizing: 'border-box',
            overflow: 'hidden', // Prevent scrollbars on the container itself
        },
    };

    useEffect(() => {
        if (!results || !results.similarity_results) {
            setGraphData({ nodes: [], links: [] });
            console.warn('Error: No Data being passed to node graph function');
            return;
        }

        const nodes = [];
        const links = [];

        nodes.push({
            id: "seed_paper",
            name: "Seed Paper",
            val: 50,
            fx: 0,
            fy: 0,
            nodeOpacity: 1
        });

        results.similarity_results.forEach((paper) => {
            nodes.push({
                id: paper.id, // Use paper.id (MUST be unique!)
                name: paper.title,
                val: 8,
            });
            links.push({
                source: 'seed_paper',
                target: paper.id, // Use paper.id
                similarity: paper.similarity_score,
            });
        });

        setGraphData({ nodes, links }); // ***UPDATE THE STATE!***

    }, [results]);

    // useEffect for customizing the force simulation (link distance/strength)
    useEffect(() => {
        if (fgRef.current && graphData.nodes.length > 0) {
           const forceLink = fgRef.current.d3Force('link');
            if (forceLink) { //Make sure forceLink is not null
                forceLink
                .distance(link => 100 * (1 - link.similarity)) // Invert similarity
                .strength(link => link.similarity * 2);
            }
        }
    }, [graphData]); // Run this effect when graphData changes

    return (
        <div style={styles.container}>
            <button onClick={toggleGraphView}>Swap</button>
            <ForceGraph2D
                ref={fgRef} // Attach the ref
                graphData={graphData}
                nodeLabel="name"
                nodeVal="val"
                linkWidth={1}
                linkDirectionalArrowLength={5}
                linkDirectionalArrowRelPos={1}
                height={400} // Set a height! (or make it responsive)
                width={600}  // Set a width! (or make it responsive)
                backgroundColor="white" // Better for 2D
                nodeColor={() => '#1f77b4'} // Default node color
                linkColor={() => '#999'} // Default link color
            />
        </div>
    );
}

export default NodeGraph;