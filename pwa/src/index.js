export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Only handle /report endpoint
    if (url.pathname === '/report') {
      if (request.method === 'OPTIONS') {
        return handleCors();
      }

      if (request.method === 'POST') {
        return handleReport(request, env);
      }
    }

    // Static assets are served before the worker is invoked by default,
    // so if a request reaches here and it's not /report, it's not found.
    return new Response('Not found', { status: 404 });
  }
};

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
    
    // Try to insert the new report
    try {
      const result = await env.DB.prepare(
        'INSERT INTO tsumego_reports (user_id, problem, success, created_at) VALUES (?, ?, ?, ?)'
      )
        .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString())
        .run();
      
      // If we get here, the insert succeeded
      return new Response('Report saved', { 
        status: 201,
        headers: { 'Access-Control-Allow-Origin': '*' }
      });
    } catch (insertError) {
      // If we get a SQLITE_CONSTRAINT error, it means the report already exists
      if (insertError.message && insertError.message.includes('UNIQUE constraint failed')) {
        return new Response('Report already exists', { 
          status: 409,
          headers: { 'Access-Control-Allow-Origin': '*' }
        });
      }
      
      // For any other error, re-throw it
      throw insertError;
    }
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { 
      status: 500,
      headers: { 'Access-Control-Allow-Origin': '*' }
    });
  }
}
