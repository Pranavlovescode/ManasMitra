'use client';

import { useState, useEffect, Suspense } from 'react';
import { SignUp } from '@clerk/nextjs';
import { useSearchParams } from 'next/navigation';
import { Heart, Shield, Users, BookOpen, TrendingUp, Clock } from 'lucide-react';

// Component that uses useSearchParams wrapped in Suspense
function SignUpContent() {
  const searchParams = useSearchParams();
  const [userRole, setUserRole] = useState('patient');

  useEffect(() => {
    const role = searchParams.get('role');
    if (role === 'therapist' || role === 'patient') {
      setUserRole(role);
    }
  }, [searchParams]);

  const roleConfig = {
    patient: {
      title: 'Join as a Patient',
      subtitle: 'Start your mental health journey with personalized care',
      color: 'from-green-400 to-blue-500',
      hoverColor: 'hover:from-green-500 hover:to-blue-600',
      features: [
        { icon: Heart, text: 'Personal mood tracking' },
        { icon: BookOpen, text: 'Private journaling space' },
        { icon: Shield, text: 'Secure & confidential' },
      ]
    },
    therapist: {
      title: 'Join as a Therapist',
      subtitle: 'Empower your practice with comprehensive patient management',
      color: 'from-purple-400 to-pink-500',
      hoverColor: 'hover:from-purple-500 hover:to-pink-600',
      features: [
        { icon: Users, text: 'Patient management dashboard' },
        { icon: TrendingUp, text: 'Progress analytics' },
        { icon: Clock, text: 'Appointment scheduling' },
      ]
    }
  };

  const config = roleConfig[userRole];

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            <span className="text-indigo-600">ManasMitra</span>
          </h1>
          <p className="text-gray-600">Mental Health Journaling Platform</p>
        </div>

        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Role Information */}
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <div className="text-center mb-8">
              <div className={`w-16 h-16 bg-linear-to-r ${config.color} rounded-full flex items-center justify-center mx-auto mb-4`}>
                {userRole === 'patient' ? (
                  <Heart className="w-8 h-8 text-white" />
                ) : (
                  <Users className="w-8 h-8 text-white" />
                )}
              </div>
              <h2 className="text-3xl font-bold text-gray-800 mb-2">{config.title}</h2>
              <p className="text-gray-600 text-lg">{config.subtitle}</p>
            </div>

            <div className="space-y-6 mb-8">
              {config.features.map((feature, index) => (
                <div key={index} className="flex items-center space-x-4">
                  <div className={`w-10 h-10 bg-linear-to-r ${config.color} rounded-full flex items-center justify-center`}>
                    <feature.icon className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-gray-700 font-medium">{feature.text}</span>
                </div>
              ))}
            </div>

            {/* Role Switch */}
            <div className="text-center border-t pt-6">
              <p className="text-gray-600 mb-4">Need a different account type?</p>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => setUserRole('patient')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    userRole === 'patient'
                      ? 'bg-green-100 text-green-700 border-2 border-green-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  Patient
                </button>
                <button
                  onClick={() => setUserRole('therapist')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    userRole === 'therapist'
                      ? 'bg-purple-100 text-purple-700 border-2 border-purple-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  Therapist
                </button>
              </div>
            </div>
          </div>

          {/* Right Side - Clerk Sign Up */}
          <div className="flex justify-center">
            <SignUp
                path="/sign-up"
                routing="path"
                signInUrl="/sign-in"
                afterSignUpUrl="/dashboard"
                appearance={{
                  elements: {
                    formButtonPrimary: `bg-linear-to-r ${config.color} ${config.hoverColor} text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg`,
                    card: 'shadow-none border-none',
                    headerTitle: 'text-2xl font-bold text-gray-800',
                    headerSubtitle: 'text-gray-600',
                    socialButtonsBlockButton: 'border-2 border-gray-200 hover:border-gray-300 transition-colors duration-200',
                    formFieldInput: 'border-2 border-gray-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200 rounded-lg',
                    footerActionLink: 'text-indigo-600 hover:text-indigo-800 font-medium',
                  },
                  variables: {
                    colorPrimary: userRole === 'patient' ? '#10b981' : '#8b5cf6',
                  }
                }}
                unsafeMetadata={{ role: userRole }}
              />
          </div>
        </div>

        {/* Bottom Links */}
        <div className="text-center mt-12">
          <p className="text-gray-600 mb-4">Already have an account?</p>
          <a 
            href="/sign-in" 
            className="text-indigo-600 hover:text-indigo-800 font-semibold underline decoration-2 underline-offset-4 hover:decoration-indigo-800 transition-colors duration-200"
          >
            Sign in here
          </a>
        </div>

        {/* Trust Indicators */}
        <div className="max-w-4xl mx-auto mt-16 text-center">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="flex flex-col items-center">
              <Shield className="w-8 h-8 text-indigo-600 mb-2" />
              <h3 className="font-semibold text-gray-800 mb-1">HIPAA Compliant</h3>
              <p className="text-sm text-gray-600">Your data is secure and private</p>
            </div>
            <div className="flex flex-col items-center">
              <Heart className="w-8 h-8 text-red-500 mb-2" />
              <h3 className="font-semibold text-gray-800 mb-1">Evidence-Based</h3>
              <p className="text-sm text-gray-600">Clinically proven methods</p>
            </div>
            <div className="flex flex-col items-center">
              <Users className="w-8 h-8 text-green-600 mb-2" />
              <h3 className="font-semibold text-gray-800 mb-1">Professional Support</h3>
              <p className="text-sm text-gray-600">Licensed therapist network</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Loading component for Suspense fallback
function SignUpLoading() {
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading sign up...</p>
      </div>
    </div>
  );
}

// Main page component with Suspense boundary
export default function SignUpPage() {
  return (
    <Suspense fallback={<SignUpLoading />}>
      <SignUpContent />
    </Suspense>
  );
}