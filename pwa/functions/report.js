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
    // Log environment information for debugging
    console.log('Environment keys:', Object.keys(env));
    
    // Parse the JSON body
    const data = await request.json();
    console.log('Received report data:', data);
    
    // Validate required fields
    if (!data.userId || !data.problem || typeof data.success !== 'boolean') {
      return new Response('Invalid request body', { 
        status: 400,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
    
    // Check if DB binding is available
    if (!env.DB) {
      console.error('D1 database binding not found. Available bindings:', Object.keys(env));
      return new Response('Database not configured', { 
        status: 500,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
    
    // Check if this report already exists
    const existingReport = await env.DB.prepare(
      'SELECT id FROM tsumego_reports WHERE user_id = ? AND problem = ?'
    )
      .bind(data.userId, data.problem)
      .first();
      
    console.log('Existing report check result:', existingReport);
    
    if (existingReport) {
      return new Response('Report already exists', { 
        status: 409,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }
    
    // Insert the new report
    const result = await env.DB.prepare(
      'INSERT INTO tsumego_reports (user_id, problem, success, created_at) VALUES (?, ?, ?, ?)'
    )
      .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString())
      .run();
    
    console.log('Insert result:', result);
    
    return new Response('Report saved', { 
      status: 201,
      headers: { 'Access-Control-Allow-Origin': '*' }
    });
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { 
      status: 500,
      headers: { 'Access-Control-Allow-Origin': '*' }
    });
  }
}