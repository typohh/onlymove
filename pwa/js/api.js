// js/api.js

// Hardcoded IP for the analysis server as requested.
// Replace <HARDCODED_IP> with the actual server IP address.
const ANALYSIS_SERVER_URL = 'http://localhost/analyze';

/**
 * Sends the current board position (as SGF) to the analysis server.
 * @param {string} sgf - The SGF string of the current position.
 * @returns {Promise<object>} A promise that resolves to the analysis result 
 *                            (e.g., { bestMove: 'Q16', score: '+2.5' }).
 */
export async function fetchAnalysis(sgf) {
    try {
        const response = await fetch(ANALYSIS_SERVER_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-go-sgf'
            },
            body: sgf
        });

        if (!response.ok) {
            throw new Error(`Server returned an error: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Failed to fetch analysis:', error);
        // Return a default error object so the calling function can handle it
        return { bestMove: null, score: 'Error' };
    }
}

