export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === '/report') {
      if (request.method === 'OPTIONS') {
        return handleCors();
      }

      if (request.method === 'POST') {
        return handleReport(request, env);
      }
    }

    // SPA fallback for navigations only. Static assets are served before
    // the worker; if no asset matches and the request looks like a page
    // navigation (e.g. /9, /13, or `/`), we serve index.html so the URL
    // stays intact for the SPA to read.
    //
    // Restricting to navigations matters: after a rebuild the bundle hash
    // changes, and a stale service-worker cache may request the *old*
    // hashed asset. Falling back to index.html for that request returns
    // HTML where JS is expected and trips strict MIME checking. For asset
    // misses we want a real 404 so the SW invalidates and re-fetches the
    // current HTML.
    if (env.ASSETS && request.method === 'GET') {
      const isNavigation = request.mode === 'navigate' ||
                           request.headers.get('sec-fetch-dest') === 'document' ||
                           (request.headers.get('accept') || '').includes('text/html');
      if (isNavigation) {
        const indexUrl = new URL('/index.html', url);
        return env.ASSETS.fetch(new Request(indexUrl, request));
      }
    }

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

    // Older clients didn't send size; default to 9 to match the historical corpus.
    const size = (data.size === 9 || data.size === 13) ? data.size : 9;

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
        'INSERT INTO tsumego_reports (user_id, problem, success, created_at, size) VALUES (?, ?, ?, ?, ?)'
      )
        .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString(), size)
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
