'use client';

import { SignIn } from '@clerk/nextjs';
import { Heart, Shield, Users, BookOpen, TrendingUp, Clock } from 'lucide-react';

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Welcome back to <span className="text-indigo-600">ManasMitra</span>
          </h1>
          <p className="text-gray-600">Continue your mental health journey</p>
        </div>

        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Welcome Message */}
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-linear-to-r from-indigo-400 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Heart className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-800 mb-2">Welcome Back</h2>
              <p className="text-gray-600 text-lg">Continue your journey towards better mental health</p>
            </div>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Patient Features */}
              <div className="bg-linear-to-br from-green-50 to-blue-50 rounded-xl p-6">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                  <Heart className="w-5 h-5 text-green-600 mr-2" />
                  For Patients
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <BookOpen className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-gray-700">Personal journaling</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-gray-700">Mood tracking</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Shield className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-gray-700">Secure & private</span>
                  </div>
                </div>
              </div>

              {/* Therapist Features */}
              <div className="bg-linear-to-br from-purple-50 to-pink-50 rounded-xl p-6">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                  <Users className="w-5 h-5 text-purple-600 mr-2" />
                  For Therapists
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Users className="w-4 h-4 text-purple-600" />
                    <span className="text-sm text-gray-700">Patient management</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="w-4 h-4 text-purple-600" />
                    <span className="text-sm text-gray-700">Progress analytics</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Clock className="w-4 h-4 text-purple-600" />
                    <span className="text-sm text-gray-700">Appointment scheduling</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="border-t pt-6">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-indigo-600">10k+</div>
                  <div className="text-sm text-gray-600">Active Users</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">500+</div>
                  <div className="text-sm text-gray-600">Licensed Therapists</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-600">1M+</div>
                  <div className="text-sm text-gray-600">Journal Entries</div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side - Clerk Sign In */}
          <div className="flex justify-center">
            {/* <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md border border-gray-100">
              
            </div> */}
            <SignIn
                path="/sign-in"
                routing="path"
                signUpUrl="/sign-up"
                afterSignInUrl="/dashboard/redirect"
                appearance={{
                  elements: {
                    formButtonPrimary: 'bg-linear-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg',
                    card: 'shadow-none border-none',
                    headerTitle: 'text-2xl font-bold text-gray-800',
                    headerSubtitle: 'text-gray-600',
                    socialButtonsBlockButton: 'border-2 border-gray-200 hover:border-gray-300 transition-colors duration-200',
                    formFieldInput: 'border-2 border-gray-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200 rounded-lg',
                    footerActionLink: 'text-indigo-600 hover:text-indigo-800 font-medium',
                    identityPreviewEditButton: 'text-indigo-600 hover:text-indigo-800',
                  },
                  variables: {
                    colorPrimary: '#6366f1',
                  }
                }}
              />
          </div>
        </div>

        {/* Bottom Links */}
        <div className="text-center mt-12">
          <p className="text-gray-600 mb-4">Don't have an account yet?</p>
          <div className="space-x-4">
            <a 
              href="/sign-up?role=patient" 
              className="inline-block bg-linear-to-r from-green-400 to-blue-500 text-white py-2 px-6 rounded-lg font-semibold hover:from-green-500 hover:to-blue-600 transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              Sign up as Patient
            </a>
            <a 
              href="/sign-up?role=therapist" 
              className="inline-block bg-linear-to-r from-purple-400 to-pink-500 text-white py-2 px-6 rounded-lg font-semibold hover:from-purple-500 hover:to-pink-600 transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              Sign up as Therapist
            </a>
          </div>
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