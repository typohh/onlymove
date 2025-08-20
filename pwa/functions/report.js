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
      return new Response(JSON.stringify({
        error: 'Invalid request body',
        receivedData: data
      }), { 
        status: 400,
        headers: { 
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }
    
    // Check if DB binding is available
    if (!env.DB) {
      return new Response(JSON.stringify({
        error: 'Database not configured',
        availableBindings: Object.keys(env)
      }), { 
        status: 500,
        headers: { 
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }
    
    // Check if this report already exists
    const existingReport = await env.DB.prepare(
      'SELECT id, user_id, problem, success, created_at FROM tsumego_reports WHERE user_id = ? AND problem = ?'
    )
      .bind(data.userId, data.problem)
      .first();
    
    // Return detailed information about existing report
    if (existingReport) {
      return new Response(JSON.stringify({
        message: 'Report already exists',
        existingReport: existingReport,
        newReportAttempt: data
      }), { 
        status: 409,
        headers: { 
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }
    
    // Insert the new report
    const result = await env.DB.prepare(
      'INSERT INTO tsumego_reports (user_id, problem, success, created_at) VALUES (?, ?, ?, ?)'
    )
      .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString())
      .run();
    
    return new Response(JSON.stringify({
      message: 'Report saved',
      insertResult: result,
      reportData: data
    }), { 
      status: 201,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Internal server error',
      errorMessage: error.message,
      errorStack: error.stack
    }), { 
      status: 500,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  }
}