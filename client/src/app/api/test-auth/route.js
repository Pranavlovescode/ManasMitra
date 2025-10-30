import { getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

// GET /api/test-auth - Test authentication
export async function GET(request) {
  try {
    console.log('Testing authentication...');
    console.log('Request URL:', request.url);
    console.log('Request headers:', Object.fromEntries(request.headers.entries()));
    
    // Try using getAuth instead of auth
    const authResult = getAuth(request);
    console.log('Auth result type:', typeof authResult);
    console.log('Auth result:', authResult);
    
    const { userId, sessionId, orgId } = authResult || {};
    console.log('Auth userId:', userId);
    console.log('Auth sessionId:', sessionId);
    console.log('Auth orgId:', orgId);
    
    if (!userId) {
      return NextResponse.json({ 
        error: 'No userId from getAuth()', 
        debug: {
          authResult,
          headers: Object.fromEntries(request.headers.entries()),
          message: 'User is not authenticated - check if user is signed in to Clerk'
        }
      }, { status: 401 });
    }

    // Try to get current user
    let user = null;
    try {
      user = await currentUser();
      console.log('Current user:', user ? 'Found' : 'Not found');
      console.log('User ID:', user?.id);
      console.log('User email:', user?.emailAddresses?.[0]?.emailAddress);
      console.log('User role:', user?.unsafeMetadata?.role);
    } catch (error) {
      console.error('Error getting current user:', error);
    }

    return NextResponse.json({
      success: true,
      authUserId: userId,
      sessionId,
      orgId,
      clerkUser: {
        id: user?.id,
        email: user?.emailAddresses?.[0]?.emailAddress,
        firstName: user?.firstName,
        lastName: user?.lastName,
        role: user?.unsafeMetadata?.role,
      },
      debug: {
        requestHeaders: Object.fromEntries(request.headers.entries()),
        authResult
      }
    });
  } catch (error) {
    console.error('Auth test error:', error);
    return NextResponse.json({ 
      error: 'Auth test failed', 
      details: error.message,
      stack: error.stack
    }, { status: 500 });
  }
}