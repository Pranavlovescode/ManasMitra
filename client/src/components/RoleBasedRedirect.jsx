'use client';

import { useUser } from '@clerk/nextjs';
import { useRouter, usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useUserProfile } from '../hooks/useUserProfile.js';
import ProfileCompletion from '../app/testing/ProfileCompletion.jsx';

export default function RoleBasedRedirect() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading, isProfileComplete } = useUserProfile();
  const router = useRouter();
  const pathname = usePathname();
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    const handleRedirect = async () => {
      console.log('[RoleBasedRedirect] useEffect triggered', { 
        isLoaded, 
        loading, 
        hasUser: !!user, 
        isRedirecting, 
        pathname,
        clerkRole: user?.publicMetadata?.role || user?.unsafeMetadata?.role,
        dbUserRole: dbUser?.role,
        isProfileComplete 
      });
      
      // Wait for Clerk to load and user to be available
      if (!isLoaded || !user) {
        console.log('[RoleBasedRedirect] Waiting for Clerk to load or user...');
        return;
      }

      // Don't wait for loading - if we have Clerk data, proceed
      if (isRedirecting) {
        console.log('[RoleBasedRedirect] Already redirecting, skipping...');
        return;
      }

      setIsRedirecting(true);

      try {
        // PRIORITY 1: Get role from Clerk metadata (most reliable)
        let userRole = user?.publicMetadata?.role || user?.unsafeMetadata?.role;

        // PRIORITY 2: Try dbUser role only if available and loading is complete
        if (!userRole && !loading && dbUser?.role) {
          console.log('[RoleBasedRedirect] Using dbUser role as fallback');
          userRole = dbUser.role;
        }

        // PRIORITY 3: Default to patient if no role found (allow login to proceed)
        if (!userRole) {
          console.log('[RoleBasedRedirect] No role found in Clerk or DB, defaulting to patient');
          userRole = 'patient';
        }
        
        console.log('[RoleBasedRedirect] Using role:', userRole, 'for user:', user.id);
        console.log('[RoleBasedRedirect] Profile complete:', isProfileComplete);

        // Only enforce profile completion on /dashboard route
        // This allows existing users to access their dashboards directly
        const shouldCheckProfile = pathname === '/dashboard' || pathname === '/dashboard/redirect';

        console.log('[RoleBasedRedirect] Profile check details:', {
          shouldCheckProfile,
          userRole,
          hasDbUser: !!dbUser,
          isProfileComplete,
          pathname
        });

        // Check if profile is complete for therapists (only if on dashboard route)
        if (shouldCheckProfile && userRole === 'therapist' && dbUser && !isProfileComplete) {
          console.log('🔄 Therapist profile incomplete, redirecting to onboarding');
          router.push('/therapist/onboarding');
          return;
        }

        // Check if profile is complete for patients (only if on dashboard route)
        if (shouldCheckProfile && userRole === 'patient' && dbUser && !isProfileComplete) {
          console.log('🔄 Patient profile incomplete, redirecting to patient details onboarding');
          router.push('/patient-details');
          return;
        }

        // Redirect based on role
        console.log(`Redirecting ${userRole} to dashboard`);
        switch (userRole) {
          case 'therapist':
            router.push('/therapist/dashboard');
            break;
          case 'admin':
            router.push('/admin/dashboard');
            break;
          case 'patient':
          default:
            router.push('/patient/dashboard');
            break;
        }
      } catch (error) {
        console.error('Error during role-based redirect:', error);
        // Don't redirect on error, let user stay where they are
      } finally {
        setIsRedirecting(false);
      }
    };

    // Add a small delay to allow Clerk to fully load
    const timeout = setTimeout(handleRedirect, 300);
    return () => clearTimeout(timeout);
  }, [user, isLoaded, dbUser, loading, router, isProfileComplete, pathname, isRedirecting]);

  // Don't show loading screen - let the redirect handle it

  // Show loading state while redirecting
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Redirecting to your dashboard...</p>
      </div>
    </div>
  );
}