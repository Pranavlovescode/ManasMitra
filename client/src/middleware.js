import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isProtectedRoute = createRouteMatcher([
  '/dashboard(.*)',
  '/api/users(.*)',
  '/api/therapists(.*)',
]);

const isPublicApiRoute = createRouteMatcher([
  '/api/webhooks(.*)',
  '/api/health',
]);

export default clerkMiddleware((auth, req) => {
  // Skip auth for public API routes (webhooks, health checks)
  if (isPublicApiRoute(req)) return;
  
  // Protect dashboard and user API routes
  if (isProtectedRoute(req)) {
    auth.protect();
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