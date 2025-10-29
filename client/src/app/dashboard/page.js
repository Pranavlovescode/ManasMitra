'use client';

import { useUser, UserButton } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useUserProfile } from '@/hooks/useUserData';
import ProfileCompletion from '@/components/ProfileCompletion';
import { Heart, Users, BookOpen, TrendingUp, Settings, LogOut } from 'lucide-react';

export default function Dashboard() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading, isProfileComplete } = useUserProfile();
  const router = useRouter();

  if (!isLoaded || loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  // Show profile completion if user exists but profile is not complete
  if (dbUser && !isProfileComplete) {
    return <ProfileCompletion />;
  }

  const userRole = user?.unsafeMetadata?.role || dbUser?.role || 'patient';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                <span className="text-indigo-600">ManasMitra</span> Dashboard
              </h1>
              <span className={`ml-4 px-3 py-1 text-xs font-medium rounded-full ${
                userRole === 'patient' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-purple-100 text-purple-800'
              }`}>
                {userRole === 'patient' ? 'Patient' : 'Therapist'}
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                Welcome, {user?.firstName || 'User'}!
              </span>
              <UserButton 
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: "h-10 w-10"
                  }
                }}
              />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {/* Welcome Section */}
          <div className="bg-white overflow-hidden shadow rounded-lg mb-6">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center">
                <div className="shrink-0">
                  {userRole === 'patient' ? (
                    <Heart className="h-8 w-8 text-green-600" />
                  ) : (
                    <Users className="h-8 w-8 text-purple-600" />
                  )}
                </div>
                <div className="ml-4">
                  <h2 className="text-lg font-medium text-gray-900">
                    {userRole === 'patient' 
                      ? 'Your Mental Health Journey' 
                      : 'Your Practice Dashboard'
                    }
                  </h2>
                  <p className="text-sm text-gray-500">
                    {userRole === 'patient' 
                      ? 'Track your progress, journal your thoughts, and connect with your therapist.'
                      : 'Manage your patients, track their progress, and provide better care.'
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
            {userRole === 'patient' ? (
              // Patient Quick Actions
              <>
                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <BookOpen className="h-6 w-6 text-blue-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">New Journal Entry</h3>
                        <p className="text-sm text-gray-500">Write about your day and feelings</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors duration-200">
                        Start Writing
                      </button>
                    </div>
                  </div>
                </div>

                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <Heart className="h-6 w-6 text-green-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Mood Check-in</h3>
                        <p className="text-sm text-gray-500">Track how you're feeling today</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors duration-200">
                        Log Mood
                      </button>
                    </div>
                  </div>
                </div>

                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <TrendingUp className="h-6 w-6 text-purple-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Progress Report</h3>
                        <p className="text-sm text-gray-500">View your mental health trends</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 transition-colors duration-200">
                        View Progress
                      </button>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              // Therapist Quick Actions
              <>
                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <Users className="h-6 w-6 text-indigo-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Patient Management</h3>
                        <p className="text-sm text-gray-500">View and manage your patients</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 transition-colors duration-200">
                        Manage Patients
                      </button>
                    </div>
                  </div>
                </div>

                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <TrendingUp className="h-6 w-6 text-green-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Analytics Dashboard</h3>
                        <p className="text-sm text-gray-500">View patient progress analytics</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors duration-200">
                        View Analytics
                      </button>
                    </div>
                  </div>
                </div>

                <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow duration-200">
                  <div className="p-6">
                    <div className="flex items-center">
                      <div className="shrink-0">
                        <Settings className="h-6 w-6 text-purple-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Practice Settings</h3>
                        <p className="text-sm text-gray-500">Configure your practice preferences</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 transition-colors duration-200">
                        Settings
                      </button>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Recent Activity */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                Recent Activity
              </h3>
              <div className="text-center py-8">
                <div className="text-gray-400 mb-2">
                  <BookOpen className="h-12 w-12 mx-auto" />
                </div>
                <p className="text-gray-500">
                  {userRole === 'patient' 
                    ? 'Start journaling to see your recent entries here.'
                    : 'Patient activities will appear here once you have active patients.'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}