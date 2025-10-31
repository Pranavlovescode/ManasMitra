'use client';

import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useUserProfile } from '../../../hooks/useUserProfile.js';

export default function DashboardIndex() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading } = useUserProfile();
  const router = useRouter();

  useEffect(() => {
    if (isLoaded && !loading) {
      // Get user role and redirect to appropriate dashboard
      const userRole = user?.unsafeMetadata?.role || user?.publicMetadata?.role || dbUser?.role || 'patient';
      
      switch (userRole) {
        case 'therapist':
          router.replace('/therapist/dashboard');
          break;
        case 'admin':
          router.replace('/admin/dashboard');
          break;
        case 'patient':
        default:
          router.replace('/patient/dashboard');
          break;
      }
    }
  }, [isLoaded, loading, user, dbUser, router]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Redirecting to your dashboard...</p>
      </div>
    </div>
  );
}