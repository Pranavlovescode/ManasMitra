import { getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    // Get authentication details
    const auth = getAuth(request);
    const { userId, sessionId } = auth;

    if (!userId) {
      return NextResponse.json({
        authenticated: false,
        error: 'No authenticated user found',
        debug: {
          auth,
          message: 'No userId found in getAuth() result'
        }
      }, { status: 401 });
    }

    // Return successful auth response
    return NextResponse.json({
      authenticated: true,
      userId,
      sessionId,
      debug: {
        auth,
        message: 'Authentication successful'
      }
    });
  } catch (error) {
    console.error('Debug Auth Error:', error);
    return NextResponse.json({
      authenticated: false,
      error: 'Internal server error during auth check',
      debug: {
        message: error.message,
        stack: error.stack
      }
    }, { status: 500 });
  }
}