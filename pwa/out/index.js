var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/index.js
var index_default = {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/report") {
      if (request.method === "OPTIONS") {
        return handleCors();
      }
      if (request.method === "POST") {
        return handleReport(request, env);
      }
    }
    return new Response("Not found", { status: 404 });
  }
};
function handleCors() {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}
__name(handleCors, "handleCors");
async function handleReport(request, env) {
  try {
    const data = await request.json();
    if (!data.userId || !data.problem || typeof data.success !== "boolean") {
      return new Response("Invalid request body", {
        status: 400,
        headers: { "Access-Control-Allow-Origin": "*" }
      });
    }
    if (!env.DB) {
      return new Response("Database not configured", {
        status: 500,
        headers: { "Access-Control-Allow-Origin": "*" }
      });
    }
    try {
      const result = await env.DB.prepare(
        "INSERT INTO tsumego_reports (user_id, problem, success, created_at) VALUES (?, ?, ?, ?)"
      ).bind(data.userId, data.problem, data.success ? 1 : 0, (/* @__PURE__ */ new Date()).toISOString()).run();
      return new Response("Report saved", {
        status: 201,
        headers: { "Access-Control-Allow-Origin": "*" }
      });
    } catch (insertError) {
      if (insertError.message && insertError.message.includes("UNIQUE constraint failed")) {
        return new Response("Report already exists", {
          status: 409,
          headers: { "Access-Control-Allow-Origin": "*" }
        });
      }
      throw insertError;
    }
  } catch (error) {
    console.error("Error handling report:", error);
    return new Response("Internal server error", {
      status: 500,
      headers: { "Access-Control-Allow-Origin": "*" }
    });
  }
}
__name(handleReport, "handleReport");
export {
  index_default as default
};
//# sourceMappingURL=index.js.map
