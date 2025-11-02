import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    // Get authentication details
    const { userId, sessionId } = await auth();

    if (!userId) {
      return NextResponse.json({
        authenticated: false,
        error: 'No authenticated user found',
        debug: {
          userId,
          sessionId,
          message: 'No userId found in auth() result'
        }
      }, { status: 401 });
    }

    // Return successful auth response
    return NextResponse.json({
      authenticated: true,
      userId,
      sessionId,
      debug: {
        userId,
        sessionId,
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