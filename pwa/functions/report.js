// Cloudflare Pages Function for handling tsumego reports
// This function receives reports from users and stores them in a D1 database

export async function onRequestPost({ request, env }) {
  // Handle CORS preflight requests
  if (request.method === 'OPTIONS') {
    return handleCors();
  }

  // Handle POST requests to /report
  return handleReport(request, env);
}

// Handle CORS headers
function handleCors() {
  return new Response(null, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

// Handle report submission
async function handleReport(request, env) {
  try {
    // Parse the JSON body
    const data = await request.json();
    
    // Validate required fields
    if (!data.userId || !data.problem || typeof data.success !== 'boolean') {
      return new Response('Invalid request body', { 
        status: 400,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
    
    // Check if DB binding is available
    if (!env.DB) {
      return new Response('Database not configured', { 
        status: 500,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
    
    // Insert the new report, ignoring if it already exists
    const result = await env.DB.prepare(
      'INSERT OR IGNORE INTO tsumego_reports (user_id, problem, success, created_at) VALUES (?, ?, ?, ?)'
    )
      .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString())
      .run();
    
    // Check if the insert was actually performed
    if (result.changes > 0) {
      return new Response('Report saved', { 
        status: 201,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    } else {
      return new Response('Report already exists', { 
        status: 409,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { 
      status: 500,
      headers: { 'Access-Control-Allow-Origin': '*' }
    });
  }
}