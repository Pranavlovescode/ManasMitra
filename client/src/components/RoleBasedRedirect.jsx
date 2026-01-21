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
        userRole: user?.publicMetadata?.role || user?.unsafeMetadata?.role,
        dbUserRole: dbUser?.role,
        isProfileComplete 
      });
      
      if (!isLoaded || loading || !user || isRedirecting) {
        console.log('[RoleBasedRedirect] Skipping redirect due to loading state');
        return;
      }

      // Only redirect if we have enough information
      if (!dbUser && !user?.publicMetadata?.role && !user?.unsafeMetadata?.role) {
        console.log('Waiting for user data before redirecting...');
        return;
      }

      setIsRedirecting(true);

      try {
        // CRITICAL: Always prioritize Clerk metadata over cached dbUser
        // This ensures we use the CURRENT session's role, not stale cached data
        let userRole = user?.publicMetadata?.role || user?.unsafeMetadata?.role;

        // Only use dbUser role if we have confirmed it matches the current Clerk user ID
        if (!userRole && dbUser && dbUser.clerkId === user.id) {
          console.log('[RoleBasedRedirect] Using dbUser role after verifying Clerk ID match');
          userRole = dbUser.role;
        }

        // If still no role, wait for database to sync instead of assuming
        // This prevents redirecting to wrong dashboard on first login
        if (!userRole) {
          console.log('[RoleBasedRedirect] No role found yet, waiting for database sync...');
          setIsRedirecting(false);
          return;
        }
        
        console.log('[RoleBasedRedirect] Using role:', userRole, 'for user:', user.id);

        // Check if profile is complete for therapists
        if (userRole === 'therapist' && !isProfileComplete) {
          // Only redirect if not already on the therapist onboarding page
          if (pathname !== '/therapist/onboarding') {
            console.log('Therapist profile incomplete, redirecting to onboarding');
            router.push('/therapist/onboarding');
          }
          return;
        }

        // Check if profile is complete for patients
        if (userRole === 'patient' && !isProfileComplete) {
          // Only redirect if not already on the patient details page
          if (pathname !== '/patient-details') {
            console.log('Patient profile incomplete, redirecting to patient details onboarding');
            router.push('/patient-details');
          }
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

    // Add a small delay to prevent immediate redirects
    const timeout = setTimeout(handleRedirect, 500);
    return () => clearTimeout(timeout);
  }, [user, isLoaded, dbUser, loading, router, isProfileComplete, pathname]);

  // Show profile completion if user exists but profile is not complete
  if (isLoaded && user && dbUser && !isProfileComplete && !isRedirecting) {
    const userRole = user?.publicMetadata?.role || user?.unsafeMetadata?.role || dbUser?.role;
    if (userRole === 'therapist') {
      // Only redirect if not already redirecting
      return (
        <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Redirecting to therapist onboarding...</p>
          </div>
        </div>
      );
    }
    if (userRole === 'patient') {
      return (
        <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Redirecting to patient onboarding...</p>
          </div>
        </div>
      );
    }
    return <ProfileCompletion />;
  }

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