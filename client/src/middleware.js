import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

const isProtectedRoute = createRouteMatcher([
  '/dashboard(.*)',
  '/patient(.*)',
  '/therapist(.*)',
  '/admin(.*)',
  '/patient-details(.*)',
  '/api/users(.*)',
  '/api/therapists(.*)',
  '/api/patients(.*)',
]);

const isPublicApiRoute = createRouteMatcher([
  '/api/webhooks(.*)',
  '/api/health',
]);

const isAuthRoute = createRouteMatcher([
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/sso-callback(.*)',
]);

const isPatientRoute = createRouteMatcher(['/patient(.*)']);
const isTherapistRoute = createRouteMatcher(['/therapist(.*)']);
const isAdminRoute = createRouteMatcher(['/admin(.*)']);

export default clerkMiddleware((auth, req) => {
  // Skip auth for public API routes (webhooks, health checks)
  if (isPublicApiRoute(req)) {
    console.log('Skipping auth for public route:', req.url);
    return;
  }

  // Skip auth for authentication routes
  if (isAuthRoute(req)) {
    console.log('Skipping auth for authentication route:', req.url);
    return;
  }
  
  // Protect dashboard and user API routes
  if (isProtectedRoute(req)) {
    console.log('Protecting route:', req.url);
    try {
      const { userId, sessionClaims } = auth.protect();
      
      // Role-based access control
      const userRole = sessionClaims?.metadata?.role || sessionClaims?.publicMetadata?.role || sessionClaims?.unsafeMetadata?.role;
      
      // Check if user is trying to access wrong dashboard
      if (isPatientRoute(req) && userRole && userRole !== 'patient') {
        return NextResponse.redirect(new URL(`/${userRole}/dashboard`, req.url));
      }
      
      if (isTherapistRoute(req) && userRole && userRole !== 'therapist') {
        return NextResponse.redirect(new URL(`/${userRole === 'admin' ? 'admin' : 'patient'}/dashboard`, req.url));
      }
      
      if (isAdminRoute(req) && userRole && userRole !== 'admin') {
        return NextResponse.redirect(new URL(`/${userRole}/dashboard`, req.url));
      }
      
      console.log('Auth protection successful for:', req.url);
    } catch (error) {
      console.error('Auth protection failed for:', req.url, error);
      throw error;
    }
  }
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
    // Include authentication routes
    '/sign-in(.*)',
    '/sign-up(.*)',
    '/sso-callback(.*)',
  ],
};