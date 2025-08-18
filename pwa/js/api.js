// js/api.js

// Analysis server endpoint
const ANALYSIS_SERVER_URL = 'https://api.onlymove.net/';

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
            // Log simple message instead of full stack trace
            console.log(`Analysis server returned ${response.status}: ${response.statusText}`);
            throw new Error(`Server returned an error: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        // Log simple message instead of full stack trace
        console.log("Analysis server unavailable or request failed");
        // Return a default error object so the calling function can handle it
        return { bestMove: null, score: 'Error' };
    }
}

