'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { Heart, Users, AlertCircle, CheckCircle } from 'lucide-react';

export default function ConfirmRolePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, isLoaded } = useUser();
  const [selectedRole, setSelectedRole] = useState('patient');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  // Get role from URL or default to patient
  useEffect(() => {
    const roleParam = searchParams.get('role');
    if (roleParam === 'therapist' || roleParam === 'patient') {
      setSelectedRole(roleParam);
    }
  }, [searchParams]);

  // If not logged in, redirect to signup
  useEffect(() => {
    if (isLoaded && !user) {
      router.push('/sign-up');
    }
  }, [isLoaded, user, router]);

  const handleConfirmRole = async () => {
    if (!user) return;

    setLoading(true);
    setError('');

    try {
      // Update user metadata in Clerk
      await user.update({
        unsafeMetadata: {
          role: selectedRole
        }
      });

      console.log(`✅ Role confirmed as ${selectedRole} in Clerk metadata`);
      
      // Try to update in database, but don't fail if it doesn't work
      try {
        const response = await fetch('/api/users', {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            role: selectedRole
          })
        });

        if (response.ok) {
          console.log('✅ Role also saved to database');
        } else {
          console.warn('⚠️ Database update failed, but Clerk metadata is set');
        }
      } catch (dbError) {
        console.warn('⚠️ Database unavailable, but Clerk metadata is set:', dbError);
      }

      setSuccess(true);

      // Redirect to dashboard after success
      setTimeout(() => {
        router.push('/dashboard');
      }, 1000);
    } catch (err) {
      console.error('Error confirming role:', err);
      setError(err.message || 'Failed to confirm role');
      setLoading(false);
    }
  };

  if (!isLoaded || !user) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
      <div className="w-full max-w-md mx-auto px-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          {success ? (
            <div className="text-center">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-800 mb-2">All Set!</h2>
              <p className="text-gray-600 mb-4">Your account is ready. Redirecting to your dashboard...</p>
            </div>
          ) : (
            <>
              <div className="text-center mb-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-2">
                  Welcome to <span className="text-indigo-600">ManasMitra</span>
                </h1>
                <p className="text-gray-600">Confirm your role to get started</p>
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-3 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              <div className="space-y-4 mb-8">
                {/* Patient Option */}
                <button
                  onClick={() => setSelectedRole('patient')}
                  className={`w-full p-6 rounded-xl border-2 transition-all duration-200 text-left ${
                    selectedRole === 'patient'
                      ? 'border-green-500 bg-green-50'
                      : 'border-gray-200 bg-white hover:border-green-300'
                  }`}
                >
                  <div className="flex items-start">
                    <Heart className={`w-6 h-6 mr-3 flex-shrink-0 mt-0.5 ${
                      selectedRole === 'patient' ? 'text-green-500' : 'text-gray-400'
                    }`} />
                    <div>
                      <h3 className="font-semibold text-gray-800">I'm a Patient</h3>
                      <p className="text-sm text-gray-600 mt-1">
                        Track your mood, journal your thoughts, and work with a therapist
                      </p>
                    </div>
                  </div>
                </button>

                {/* Therapist Option */}
                <button
                  onClick={() => setSelectedRole('therapist')}
                  className={`w-full p-6 rounded-xl border-2 transition-all duration-200 text-left ${
                    selectedRole === 'therapist'
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 bg-white hover:border-purple-300'
                  }`}
                >
                  <div className="flex items-start">
                    <Users className={`w-6 h-6 mr-3 flex-shrink-0 mt-0.5 ${
                      selectedRole === 'therapist' ? 'text-purple-500' : 'text-gray-400'
                    }`} />
                    <div>
                      <h3 className="font-semibold text-gray-800">I'm a Therapist</h3>
                      <p className="text-sm text-gray-600 mt-1">
                        Manage your practice, monitor patient progress, and provide care
                      </p>
                    </div>
                  </div>
                </button>
              </div>

              <button
                onClick={handleConfirmRole}
                disabled={loading}
                className="w-full bg-linear-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg disabled:hover:scale-100"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Confirming...
                  </span>
                ) : (
                  'Confirm Role & Continue'
                )}
              </button>

              <p className="text-center text-sm text-gray-500 mt-6">
                You can change your role later in your account settings
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
