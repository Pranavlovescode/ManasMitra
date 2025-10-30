import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isProtectedRoute = createRouteMatcher([
  '/dashboard(.*)',
  '/patient-details(.*)',
  '/api/users(.*)',
  '/api/therapists(.*)',
  '/api/patients(.*)',
]);

const isPublicApiRoute = createRouteMatcher([
  '/api/webhooks(.*)',
  '/api/health',
]);

export default clerkMiddleware((auth, req) => {
  // Skip auth for public API routes (webhooks, health checks)
  if (isPublicApiRoute(req)) {
    console.log('Skipping auth for public route:', req.url);
    return;
  }
  
  // Protect dashboard and user API routes
  if (isProtectedRoute(req)) {
    console.log('Protecting route:', req.url);
    try {
      auth.protect();
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
  ],
};