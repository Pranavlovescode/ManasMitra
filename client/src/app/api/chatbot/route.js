// Proxy route: POST /api/generate -> forwards to backend /generate
// This lets the frontend call the backend without CORS issues and centralizes the backend URL.

export async function POST(req) {
  try {
    const body = await req.json();

    // Use environment variable if available (set NEXT_PUBLIC_BACKEND_URL in dev/prod)
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_API_URL || 'http://localhost:8000';

    const res = await fetch(`${backend}/chatbot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    return new Response(JSON.stringify(data), {
      status: res.status,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

export async function GET() {
  // Simple message to confirm the API route is reachable from the frontend
  return new Response(JSON.stringify({ status: 'generate proxy ok' }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}
