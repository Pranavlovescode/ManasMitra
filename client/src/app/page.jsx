import Features from '../components/Features';

export default function Home() {
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-800 mb-6">
            Welcome to <span className="text-indigo-600">ManasMitra</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Your trusted companion for mental health journaling and therapy management. 
            Connect patients and therapists in a secure, supportive environment.
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-semibold text-center text-gray-700 mb-12">
            Choose Your Role
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            {/* Patient Card */}
            <div className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100">
              <div className="text-center">
                <div className="w-20 h-20 bg-linear-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-4">I'm a Patient</h3>
                <p className="text-gray-600 mb-8 leading-relaxed">
                  Start your mental health journey with personalized journaling, mood tracking, 
                  and connect with your therapist for better care.
                </p>
                <div className="space-y-4 mb-8">
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Daily mood tracking
                  </div>
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Personal journaling
                  </div>
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Therapist connection
                  </div>
                </div>
                <a 
                  href="/sign-up?role=patient" 
                  className="w-full bg-linear-to-r from-green-400 to-blue-500 text-white py-3 px-6 rounded-lg font-semibold hover:from-green-500 hover:to-blue-600 transition-all duration-200 transform hover:scale-105 shadow-lg block text-center"
                >
                  Get Started as Patient
                </a>
              </div>
            </div>

            {/* Therapist Card */}
            <div className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100">
              <div className="text-center">
                <div className="w-20 h-20 bg-linear-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-4">I'm a Therapist</h3>
                <p className="text-gray-600 mb-8 leading-relaxed">
                  Manage your practice efficiently, monitor patient progress, 
                  and provide better care with comprehensive tools.
                </p>
                <div className="space-y-4 mb-8">
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-purple-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Patient management
                  </div>
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-purple-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Progress tracking
                  </div>
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 text-purple-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    Analytics dashboard
                  </div>
                </div>
                <a 
                  href="/sign-up?role=therapist" 
                  className="w-full bg-linear-to-r from-purple-400 to-pink-500 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-500 hover:to-pink-600 transition-all duration-200 transform hover:scale-105 shadow-lg block text-center"
                >
                  Get Started as Therapist
                </a>
              </div>
            </div>
          </div>

          <div className="text-center mt-12">
            <p className="text-gray-600 mb-4">Already have an account?</p>
            <a 
              href="/sign-in" 
              className="text-indigo-600 hover:text-indigo-800 font-semibold underline decoration-2 underline-offset-4 hover:decoration-indigo-800 transition-colors duration-200"
            >
              Sign in here
            </a>
          </div>
        </div>
      </div>
      
      {/* Features Section */}
      <Features />
    </div>
  );
}
