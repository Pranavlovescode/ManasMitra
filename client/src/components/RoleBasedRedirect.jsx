'use client';

import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useUserProfile } from '../hooks/useUserProfile.js';
import ProfileCompletion from '../app/testing/ProfileCompletion.jsx';

export default function RoleBasedRedirect() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading, isProfileComplete } = useUserProfile();
  const router = useRouter();
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    const handleRedirect = async () => {
      if (!isLoaded || loading || !user || isRedirecting) return;

      // Only redirect if we have enough information
      if (!dbUser && !user?.publicMetadata?.role && !user?.unsafeMetadata?.role) {
        console.log('Waiting for user data before redirecting...');
        return;
      }

      setIsRedirecting(true);

      try {
        // First, try to get user role from Clerk metadata
        let userRole = user?.publicMetadata?.role || user?.unsafeMetadata?.role;

        // If no role in Clerk metadata, check our database
        if (!userRole && dbUser) {
          userRole = dbUser.role;
        }

        // If still no role, try fetching from API
        if (!userRole) {
          try {
            const response = await fetch('/api/users', {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            });

            if (response.ok) {
              const userData = await response.json();
              userRole = userData?.role;
            }
          } catch (error) {
            console.error('Error fetching user data:', error);
          }
        }

        // If user exists but no role is set, redirect to role selection or profile completion
        if (!userRole) {
          console.log('No role found, redirecting to sign-up for role selection');
          router.push('/sign-up');
          return;
        }

        // Check if profile is complete for therapists
        if (userRole === 'therapist' && !isProfileComplete) {
          console.log('Therapist profile incomplete, redirecting to onboarding');
          router.push('/therapist/onboarding');
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
  }, [user, isLoaded, dbUser, loading, router, isProfileComplete]);

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