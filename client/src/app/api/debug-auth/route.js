import { getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

// Simple debug endpoint to test auth
export async function GET(request) {
  try {
    const authResult = getAuth(request);
    const { userId } = authResult || {};
    
    return NextResponse.json({
      authenticated: !!userId,
      userId: userId || null,
      authResult,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return NextResponse.json({
      error: 'Auth check failed',
      details: error.message
    }, { status: 500 });
  }
}

export async function POST(request) {
  try {
    const authResult = getAuth(request);
    const { userId } = authResult || {};
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    return NextResponse.json({
      success: true,
      message: 'POST request authenticated successfully',
      userId,
      authResult
    });
  } catch (error) {
    return NextResponse.json({
      error: 'Auth check failed',
      details: error.message
    }, { status: 500 });
  }
}