"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@clerk/nextjs";
import { useUserProfile } from "../../../hooks/useUserProfile";
import {
  User,
  Stethoscope,
  Phone,
  MapPin,
  FileText,
  Award,
  GraduationCap,
  Clock,
  Users,
  Building,
  CreditCard,
  Camera,
  CheckCircle,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";

const ONBOARDING_STEPS = [
  { id: 1, title: "Basic Information", icon: User },
  { id: 2, title: "Professional Details", icon: Stethoscope },
  { id: 3, title: "Education & Credentials", icon: GraduationCap },
  { id: 4, title: "Practice Information", icon: Building },
  { id: 5, title: "Preferences", icon: Clock },
  { id: 6, title: "Review & Complete", icon: CheckCircle },
];

export default function TherapistOnboarding() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading: profileLoading } = useUserProfile();
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [lastSaved, setLastSaved] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);
  const router = useRouter();

  // Storage key for persistence
  const STORAGE_KEY = 'therapist-onboarding-data';
  const STEP_STORAGE_KEY = 'therapist-onboarding-step';

  // Redirect if not a therapist
  useEffect(() => {
    if (user && isLoaded) {
      const userRole = user?.publicMetadata?.role || user?.unsafeMetadata?.role;
      if (userRole && userRole !== "therapist") {
        console.log(`Non-therapist (${userRole}) accessing onboarding, redirecting to dashboard`);
        router.push("/dashboard");
      }
    }
  }, [user, router, isLoaded]);

  // Function to get initial form data from localStorage or defaults
  const getInitialFormData = () => {
    if (typeof window === 'undefined') return getDefaultFormData();
    
    try {
      const savedData = localStorage.getItem(STORAGE_KEY);
      if (savedData) {
        const parsedData = JSON.parse(savedData);
        // Merge with defaults to ensure all fields exist
        return {
          ...getDefaultFormData(),
          ...parsedData,
          // Always use latest user info from Clerk
          firstName: user?.firstName || parsedData.firstName || "",
          lastName: user?.lastName || parsedData.lastName || "",
          email: user?.emailAddresses?.[0]?.emailAddress || parsedData.email || "",
        };
      }
    } catch (error) {
      console.error('Error loading saved form data:', error);
    }
    
    return getDefaultFormData();
  };

  // Function to get default form data
  const getDefaultFormData = () => ({
    // Basic Information
    firstName: user?.firstName || "",
    lastName: user?.lastName || "",
    email: user?.emailAddresses?.[0]?.emailAddress || "",
    phone: "",
    dateOfBirth: "",

    // Address
    address: {
      street: "",
      city: "",
      state: "",
      zipCode: "",
      country: "US",
    },

    // Professional Details
    licenseNumber: "",
    licenseState: "",
    licenseExpiry: "",
    yearsOfExperience: "",
    specializations: [],
    therapyApproaches: [],
    languages: ["English"],

    // Education & Credentials
    education: [],
    certifications: [],

    // Practice Information
    practiceType: "", // private, group, hospital, community
    practiceName: "",
    officeAddress: {
      street: "",
      city: "",
      state: "",
      zipCode: "",
      country: "US",
    },
    insurance: [],
    fees: {
      individual: "",
      couple: "",
      group: "",
      initialConsultation: "",
    },

    // Preferences
    sessionFormats: [], // in-person, virtual, both
    availabilityHours: [],
    acceptingNewPatients: true,
    emergencyContact: {
      name: "",
      phone: "",
      relationship: "",
    },

    // Professional Statement
    bio: "",
    profilePhoto: null,
  });

  const [formData, setFormData] = useState(getInitialFormData);

  // Load saved step and check for saved data on component mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedStep = localStorage.getItem(STEP_STORAGE_KEY);
      const savedData = localStorage.getItem(STORAGE_KEY);
      
      if (savedStep) {
        try {
          const step = parseInt(savedStep, 10);
          if (step >= 1 && step <= ONBOARDING_STEPS.length) {
            setCurrentStep(step);
          }
        } catch (error) {
          console.error('Error loading saved step:', error);
        }
      }
      
      if (savedData) {
        setDataLoaded(true);
        // Hide the notification after 5 seconds
        setTimeout(() => setDataLoaded(false), 5000);
      }
    }
  }, []);

  // Save form data to localStorage whenever it changes (debounced)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saveTimer = setTimeout(() => {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(formData));
          setLastSaved(new Date());
        } catch (error) {
          console.error('Error saving form data:', error);
        }
      }, 500); // Debounce by 500ms

      return () => clearTimeout(saveTimer);
    }
  }, [formData]);

  // Save current step to localStorage whenever it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem(STEP_STORAGE_KEY, currentStep.toString());
      } catch (error) {
        console.error('Error saving current step:', error);
      }
    }
  }, [currentStep]);

  // Warn user before leaving page if there's unsaved data
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (typeof window !== 'undefined') {
        const savedData = localStorage.getItem(STORAGE_KEY);
        if (savedData && !success) {
          e.preventDefault();
          e.returnValue = '';
          return '';
        }
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [success]);



  // Function to clear saved data (called on successful submission)
  const clearSavedData = () => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(STEP_STORAGE_KEY);
      } catch (error) {
        console.error('Error clearing saved data:', error);
      }
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    if (name.includes(".")) {
      const [parent, child] = name.split(".");
      setFormData((prev) => ({
        ...prev,
        [parent]: {
          ...prev[parent],
          [child]: value,
        },
      }));
    } else {
      setFormData((prev) => ({
        ...prev,
        [name]: value,
      }));
    }
  };

  const handleArrayInput = (field, value) => {
    const items = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item);
    setFormData((prev) => ({
      ...prev,
      [field]: items,
    }));
  };

  const handleMultiSelect = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter((item) => item !== value)
        : [...prev[field], value],
    }));
  };

  const addEducation = () => {
    setFormData((prev) => ({
      ...prev,
      education: [
        ...prev.education,
        { degree: "", institution: "", year: "", field: "" },
      ],
    }));
  };

  const updateEducation = (index, field, value) => {
    setFormData((prev) => ({
      ...prev,
      education: prev.education.map((edu, i) =>
        i === index ? { ...edu, [field]: value } : edu
      ),
    }));
  };

  const removeEducation = (index) => {
    setFormData((prev) => ({
      ...prev,
      education: prev.education.filter((_, i) => i !== index),
    }));
  };

  const nextStep = () => {
    if (currentStep < ONBOARDING_STEPS.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Only allow submission on the final step
    if (currentStep !== ONBOARDING_STEPS.length) {
      console.log('Form submission blocked - not on final step');
      return;
    }
    
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const therapistData = {
        professionalInfo: {
          licenseNumber: formData.licenseNumber,
          licenseState: formData.licenseState,
          licenseExpiry: formData.licenseExpiry,
          yearsOfExperience: formData.yearsOfExperience
            ? parseInt(formData.yearsOfExperience)
            : 0,
          specializations: formData.specializations,
          therapyApproaches: formData.therapyApproaches,
          languages: formData.languages,
          education: formData.education.filter(
            (edu) => edu.degree && edu.institution
          ),
          certifications: formData.certifications,
        },
        contactInfo: {
          phoneNumber: formData.phone,
          officeAddress: formData.officeAddress,
        },
        preferences: {
          sessionFormats: formData.sessionFormats,
          acceptingNewPatients: formData.acceptingNewPatients,
          availabilityHours: formData.availabilityHours,
        },
        practiceInfo: {
          practiceType: formData.practiceType,
          practiceName: formData.practiceName,
          insurance: formData.insurance,
          fees: formData.fees,
        },
        personalInfo: {
          bio: formData.bio,
          dateOfBirth: formData.dateOfBirth,
          address: formData.address,
          emergencyContact: formData.emergencyContact,
        },
      };

      console.log(
        "📤 Sending comprehensive therapist profile data:",
        therapistData
      );

      const response = await fetch("/api/therapists/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(therapistData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to complete profile");
      }

      setSuccess(
        "Profile completed successfully! Redirecting to your dashboard..."
      );

      // Clear saved data since profile is completed
      clearSavedData();

      // Use a shorter delay and more reliable redirect
      console.log("🔄 Profile completed, redirecting to dashboard...");
      setTimeout(() => {
        router.push("/therapist/dashboard").catch((error) => {
          console.error("❌ Router redirect failed:", error);
          // Fallback to window.location
          window.location.href = "/therapist/dashboard";
        });
      }, 1500);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (profileLoading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mx-auto w-20 h-20 bg-linear-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center mb-6">
            <Stethoscope className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Welcome to <span className="text-indigo-600">ManasMitra</span>
          </h1>
          <p className="text-xl text-gray-600 mb-4">
            Complete your professional profile to get started
          </p>
          
          {/* Auto-save indicator and controls */}
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="flex items-center text-sm text-green-600 bg-green-50 px-3 py-1 rounded-full">
              <CheckCircle className="w-4 h-4 mr-2" />
              {lastSaved ? (
                <span>
                  Saved at {lastSaved.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              ) : (
                <span>Auto-saving progress</span>
              )}
            </div>
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all saved data? This action cannot be undone.')) {
                  clearSavedData();
                  setFormData(getDefaultFormData());
                  setCurrentStep(1);
                  setLastSaved(null);
                }
              }}
              className="text-sm text-gray-500 hover:text-red-600 underline transition-colors duration-200"
            >
              Clear saved data
            </button>
          </div>

          {/* Progress Bar */}
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center justify-between mb-8">
              {ONBOARDING_STEPS.map((step, index) => (
                <div key={step.id} className="flex flex-col items-center">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-200 ${
                      currentStep >= step.id
                        ? "bg-indigo-600 border-indigo-600 text-white"
                        : "bg-white border-gray-300 text-gray-400"
                    }`}
                  >
                    <step.icon className="w-5 h-5" />
                  </div>
                  <p
                    className={`text-xs mt-2 font-medium ${
                      currentStep >= step.id
                        ? "text-indigo-600"
                        : "text-gray-400"
                    }`}
                  >
                    {step.title}
                  </p>
                  {index < ONBOARDING_STEPS.length - 1 && (
                    <div
                      className={`hidden md:block w-16 h-0.5 mt-6 -mr-16 ${
                        currentStep > step.id ? "bg-indigo-600" : "bg-gray-300"
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Form Card */}
        <div className="bg-white shadow-xl rounded-2xl border border-gray-100">
          <div className="px-8 py-8">
            {/* Success/Error Messages */}
            {dataLoaded && (
              <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center">
                  <CheckCircle className="w-5 h-5 text-blue-600 mr-2" />
                  <p className="text-blue-700 font-medium">
                    Previous progress restored! You can continue from where you left off.
                  </p>
                </div>
              </div>
            )}

            {success && (
              <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                    <p className="text-green-700 font-medium">{success}</p>
                  </div>
                  <button
                    onClick={() => {
                      console.log("🔄 Manual redirect triggered");
                      window.location.href = "/therapist/dashboard";
                    }}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors duration-200 text-sm font-medium"
                  >
                    Go to Dashboard
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit}>
              {/* Step 1: Basic Information */}
              {currentStep === 1 && (
                <BasicInformationStep
                  formData={formData}
                  onChange={handleInputChange}
                />
              )}

              {/* Step 2: Professional Details */}
              {currentStep === 2 && (
                <ProfessionalDetailsStep
                  formData={formData}
                  onChange={handleInputChange}
                  onArrayChange={handleArrayInput}
                  onMultiSelect={handleMultiSelect}
                />
              )}

              {/* Step 3: Education & Credentials */}
              {currentStep === 3 && (
                <EducationCredentialsStep
                  formData={formData}
                  onChange={handleInputChange}
                  onAddEducation={addEducation}
                  onUpdateEducation={updateEducation}
                  onRemoveEducation={removeEducation}
                  onArrayChange={handleArrayInput}
                />
              )}

              {/* Step 4: Practice Information */}
              {currentStep === 4 && (
                <PracticeInformationStep
                  formData={formData}
                  onChange={handleInputChange}
                  onArrayChange={handleArrayInput}
                  onMultiSelect={handleMultiSelect}
                />
              )}

              {/* Step 5: Preferences */}
              {currentStep === 5 && (
                <PreferencesStep
                  formData={formData}
                  onChange={handleInputChange}
                  onMultiSelect={handleMultiSelect}
                />
              )}

              {/* Step 6: Review & Complete */}
              {currentStep === 6 && (
                <ReviewCompleteStep
                  formData={formData}
                  onChange={handleInputChange}
                />
              )}

              {/* Navigation Buttons */}
              <div className="flex justify-between pt-8 border-t border-gray-200 mt-8">
                <button
                  type="button"
                  onClick={prevStep}
                  disabled={currentStep === 1}
                  className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Previous
                </button>

                {currentStep < ONBOARDING_STEPS.length ? (
                  <button
                    type="button"
                    onClick={nextStep}
                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors duration-200 flex items-center"
                  >
                    Next
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={loading}
                    className="px-8 py-3 bg-linear-to-r from-purple-400 to-pink-500 text-white rounded-lg font-medium hover:from-purple-500 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center transition-all duration-200 transform hover:scale-105 shadow-lg"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                        Completing Profile...
                      </>
                    ) : (
                      <>
                        Complete Profile
                        <CheckCircle className="w-5 h-5 ml-2" />
                      </>
                    )}
                  </button>
                )}
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

// Step Components
function BasicInformationStep({ formData, onChange }) {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <User className="w-12 h-12 text-indigo-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">Basic Information</h2>
        <p className="text-gray-600">Let's start with your personal details</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            First Name *
          </label>
          <input
            type="text"
            name="firstName"
            value={formData.firstName}
            onChange={onChange}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Enter your first name"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Last Name *
          </label>
          <input
            type="text"
            name="lastName"
            value={formData.lastName}
            onChange={onChange}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Enter your last name"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Phone className="w-4 h-4 inline mr-2" />
            Phone Number *
          </label>
          <input
            type="tel"
            name="phone"
            value={formData.phone}
            onChange={onChange}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="(555) 123-4567"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Date of Birth
          </label>
          <input
            type="date"
            name="dateOfBirth"
            value={formData.dateOfBirth}
            onChange={onChange}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
      </div>

      {/* Address Section */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-900 flex items-center">
          <MapPin className="w-5 h-5 mr-2" />
          Personal Address
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="md:col-span-2">
            <input
              type="text"
              name="address.street"
              value={formData.address.street}
              onChange={onChange}
              placeholder="Street Address"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="address.city"
              value={formData.address.city}
              onChange={onChange}
              placeholder="City"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="address.state"
              value={formData.address.state}
              onChange={onChange}
              placeholder="State"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="address.zipCode"
              value={formData.address.zipCode}
              onChange={onChange}
              placeholder="ZIP Code"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ProfessionalDetailsStep({
  formData,
  onChange,
  onArrayChange,
  onMultiSelect,
}) {
  const commonSpecializations = [
    "Anxiety Disorders",
    "Depression",
    "PTSD",
    "Bipolar Disorder",
    "ADHD",
    "Eating Disorders",
    "Substance Abuse",
    "Family Therapy",
    "Couples Therapy",
    "Child Psychology",
    "Adolescent Psychology",
    "Grief Counseling",
    "Stress Management",
  ];

  const therapyApproaches = [
    "Cognitive Behavioral Therapy (CBT)",
    "Dialectical Behavior Therapy (DBT)",
    "Psychoanalytic Therapy",
    "Humanistic Therapy",
    "Solution-Focused Therapy",
    "EMDR",
    "Mindfulness-Based Therapy",
    "Art Therapy",
    "Play Therapy",
  ];

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <Stethoscope className="w-12 h-12 text-indigo-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">
          Professional Details
        </h2>
        <p className="text-gray-600">
          Tell us about your professional credentials
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <FileText className="w-4 h-4 inline mr-2" />
            License Number *
          </label>
          <input
            type="text"
            name="licenseNumber"
            value={formData.licenseNumber}
            onChange={onChange}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Professional License Number"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            License State *
          </label>
          <input
            type="text"
            name="licenseState"
            value={formData.licenseState}
            onChange={onChange}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="State where licensed"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            License Expiry Date
          </label>
          <input
            type="date"
            name="licenseExpiry"
            value={formData.licenseExpiry}
            onChange={onChange}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Clock className="w-4 h-4 inline mr-2" />
            Years of Experience *
          </label>
          <input
            type="number"
            name="yearsOfExperience"
            value={formData.yearsOfExperience}
            onChange={onChange}
            required
            min="0"
            max="50"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Years of experience"
          />
        </div>
      </div>

      {/* Specializations */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          <Award className="w-4 h-4 inline mr-2" />
          Specializations *
        </label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          {commonSpecializations.map((spec) => (
            <label key={spec} className="flex items-center">
              <input
                type="checkbox"
                checked={formData.specializations.includes(spec)}
                onChange={() => onMultiSelect("specializations", spec)}
                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 mr-2"
              />
              <span className="text-sm text-gray-700">{spec}</span>
            </label>
          ))}
        </div>
        <input
          type="text"
          onChange={(e) => onArrayChange("specializations", e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="Add custom specializations (separate with commas)"
        />
      </div>

      {/* Therapy Approaches */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Therapy Approaches
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {therapyApproaches.map((approach) => (
            <label key={approach} className="flex items-center">
              <input
                type="checkbox"
                checked={formData.therapyApproaches.includes(approach)}
                onChange={() => onMultiSelect("therapyApproaches", approach)}
                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 mr-2"
              />
              <span className="text-sm text-gray-700">{approach}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Languages */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Languages Spoken
        </label>
        <input
          type="text"
          onChange={(e) => onArrayChange("languages", e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="e.g., English, Spanish, French (separate with commas)"
          defaultValue="English"
        />
      </div>
    </div>
  );
}

function EducationCredentialsStep({
  formData,
  onChange,
  onAddEducation,
  onUpdateEducation,
  onRemoveEducation,
  onArrayChange,
}) {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <GraduationCap className="w-12 h-12 text-indigo-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">
          Education & Credentials
        </h2>
        <p className="text-gray-600">
          Share your educational background and certifications
        </p>
      </div>

      {/* Education */}
      <div>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium text-gray-900">Education</h3>
          <button
            type="button"
            onClick={onAddEducation}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors duration-200"
          >
            Add Education
          </button>
        </div>

        {formData.education.map((edu, index) => (
          <div key={index} className="bg-gray-50 p-4 rounded-lg mb-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <input
                  type="text"
                  value={edu.degree}
                  onChange={(e) =>
                    onUpdateEducation(index, "degree", e.target.value)
                  }
                  placeholder="Degree (e.g., Ph.D., M.A., Psy.D.)"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <input
                  type="text"
                  value={edu.institution}
                  onChange={(e) =>
                    onUpdateEducation(index, "institution", e.target.value)
                  }
                  placeholder="Institution Name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <input
                  type="text"
                  value={edu.field}
                  onChange={(e) =>
                    onUpdateEducation(index, "field", e.target.value)
                  }
                  placeholder="Field of Study"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div className="flex">
                <input
                  type="number"
                  value={edu.year}
                  onChange={(e) =>
                    onUpdateEducation(index, "year", e.target.value)
                  }
                  placeholder="Graduation Year"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  type="button"
                  onClick={() => onRemoveEducation(index)}
                  className="ml-2 px-3 py-2 bg-red-100 text-red-600 rounded-md hover:bg-red-200"
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        ))}

        {formData.education.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <GraduationCap className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>
              No education entries yet. Click "Add Education" to get started.
            </p>
          </div>
        )}
      </div>

      {/* Certifications */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          <Award className="w-4 h-4 inline mr-2" />
          Additional Certifications
        </label>
        <input
          type="text"
          onChange={(e) => onArrayChange("certifications", e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="e.g., Certified EMDR Therapist, Board Certified (separate with commas)"
        />
        <p className="text-sm text-gray-500 mt-1">
          Separate multiple certifications with commas
        </p>
      </div>
    </div>
  );
}

function PracticeInformationStep({
  formData,
  onChange,
  onArrayChange,
  onMultiSelect,
}) {
  const practiceTypes = [
    { value: "private", label: "Private Practice" },
    { value: "group", label: "Group Practice" },
    { value: "hospital", label: "Hospital/Medical Center" },
    { value: "community", label: "Community Mental Health Center" },
    { value: "other", label: "Other" },
  ];

  const commonInsurance = [
    "Aetna",
    "Blue Cross Blue Shield",
    "Cigna",
    "UnitedHealth",
    "Humana",
    "Medicare",
    "Medicaid",
    "Kaiser Permanente",
    "Anthem",
  ];

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <Building className="w-12 h-12 text-indigo-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">
          Practice Information
        </h2>
        <p className="text-gray-600">
          Tell us about your practice and services
        </p>
      </div>

      {/* Practice Type */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Practice Type *
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {practiceTypes.map((type) => (
            <label
              key={type.value}
              className="flex items-center p-3 border border-gray-300 rounded-lg hover:border-indigo-500 cursor-pointer"
            >
              <input
                type="radio"
                name="practiceType"
                value={type.value}
                checked={formData.practiceType === type.value}
                onChange={onChange}
                className="text-indigo-600 focus:ring-indigo-500 mr-3"
              />
              <span className="text-gray-700">{type.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Practice Name */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Practice/Organization Name
        </label>
        <input
          type="text"
          name="practiceName"
          value={formData.practiceName}
          onChange={onChange}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="Enter practice or organization name"
        />
      </div>

      {/* Office Address */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-900 flex items-center">
          <MapPin className="w-5 h-5 mr-2" />
          Office Address
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="md:col-span-2">
            <input
              type="text"
              name="officeAddress.street"
              value={formData.officeAddress.street}
              onChange={onChange}
              placeholder="Office Street Address"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="officeAddress.city"
              value={formData.officeAddress.city}
              onChange={onChange}
              placeholder="City"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="officeAddress.state"
              value={formData.officeAddress.state}
              onChange={onChange}
              placeholder="State"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="text"
              name="officeAddress.zipCode"
              value={formData.officeAddress.zipCode}
              onChange={onChange}
              placeholder="ZIP Code"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>
      </div>

      {/* Insurance */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          <CreditCard className="w-4 h-4 inline mr-2" />
          Insurance Accepted
        </label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          {commonInsurance.map((insurance) => (
            <label key={insurance} className="flex items-center">
              <input
                type="checkbox"
                checked={formData.insurance.includes(insurance)}
                onChange={() => onMultiSelect("insurance", insurance)}
                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 mr-2"
              />
              <span className="text-sm text-gray-700">{insurance}</span>
            </label>
          ))}
        </div>
        <input
          type="text"
          onChange={(e) => onArrayChange("insurance", e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="Add other insurance plans (separate with commas)"
        />
      </div>

      {/* Fees */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Session Fees (Optional)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Individual Session
            </label>
            <input
              type="number"
              name="fees.individual"
              value={formData.fees.individual}
              onChange={onChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="$"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Couples Session
            </label>
            <input
              type="number"
              name="fees.couple"
              value={formData.fees.couple}
              onChange={onChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="$"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Group Session
            </label>
            <input
              type="number"
              name="fees.group"
              value={formData.fees.group}
              onChange={onChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="$"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Initial Consultation
            </label>
            <input
              type="number"
              name="fees.initialConsultation"
              value={formData.fees.initialConsultation}
              onChange={onChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="$"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function PreferencesStep({ formData, onChange, onMultiSelect }) {
  const sessionFormats = [
    { value: "in-person", label: "In-Person Sessions", icon: Users },
    { value: "virtual", label: "Virtual/Telehealth Sessions", icon: Camera },
    { value: "both", label: "Both In-Person & Virtual", icon: Clock },
  ];

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <Clock className="w-12 h-12 text-indigo-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">
          Preferences & Availability
        </h2>
        <p className="text-gray-600">
          Set your practice preferences and availability
        </p>
      </div>

      {/* Session Formats */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Session Formats *
        </label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {sessionFormats.map((format) => (
            <label
              key={format.value}
              className="flex items-center p-4 border border-gray-300 rounded-lg hover:border-indigo-500 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={formData.sessionFormats.includes(format.value)}
                onChange={() => onMultiSelect("sessionFormats", format.value)}
                className="text-indigo-600 focus:ring-indigo-500 mr-3"
              />
              <div>
                <format.icon className="w-6 h-6 text-indigo-600 mb-2" />
                <span className="text-gray-700 font-medium">
                  {format.label}
                </span>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Accepting New Patients */}
      <div>
        <label className="flex items-center p-4 border border-gray-300 rounded-lg">
          <input
            type="checkbox"
            name="acceptingNewPatients"
            checked={formData.acceptingNewPatients}
            onChange={(e) =>
              onChange({
                target: {
                  name: "acceptingNewPatients",
                  value: e.target.checked,
                },
              })
            }
            className="text-indigo-600 focus:ring-indigo-500 mr-3"
          />
          <div>
            <span className="text-gray-700 font-medium">
              Currently Accepting New Patients
            </span>
            <p className="text-sm text-gray-500">
              Check this if you're currently taking on new clients
            </p>
          </div>
        </label>
      </div>

      {/* Emergency Contact */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-900">Emergency Contact</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <input
              type="text"
              name="emergencyContact.name"
              value={formData.emergencyContact.name}
              onChange={onChange}
              placeholder="Emergency Contact Name"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div>
            <input
              type="tel"
              name="emergencyContact.phone"
              value={formData.emergencyContact.phone}
              onChange={onChange}
              placeholder="Emergency Contact Phone"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          <div className="md:col-span-2">
            <input
              type="text"
              name="emergencyContact.relationship"
              value={formData.emergencyContact.relationship}
              onChange={onChange}
              placeholder="Relationship (e.g., Spouse, Colleague, Supervisor)"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ReviewCompleteStep({ formData, onChange }) {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900">Review & Complete</h2>
        <p className="text-gray-600">
          Add a professional bio and review your information
        </p>
      </div>

      {/* Professional Bio */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Professional Bio
        </label>
        <textarea
          name="bio"
          value={formData.bio}
          onChange={onChange}
          onKeyDown={(e) => {
            // Prevent Enter key from submitting the form
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
            }
          }}
          rows={6}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="Write a brief professional bio that will be visible to patients. Include your approach to therapy, areas of expertise, and what patients can expect when working with you."
        />
        <p className="text-sm text-gray-500 mt-1">
          This will be displayed on your profile to help patients understand
          your approach and expertise.
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-indigo-50 p-4 rounded-lg">
          <h3 className="font-semibold text-indigo-900 mb-3">
            Professional Info
          </h3>
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-gray-600">License:</span>{" "}
              {formData.licenseNumber || "Not provided"}
            </p>
            <p>
              <span className="text-gray-600">Experience:</span>{" "}
              {formData.yearsOfExperience || "Not provided"} years
            </p>
            <p>
              <span className="text-gray-600">Specializations:</span>{" "}
              {formData.specializations.length || 0} selected
            </p>
          </div>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="font-semibold text-purple-900 mb-3">
            Practice Details
          </h3>
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-gray-600">Type:</span>{" "}
              {formData.practiceType || "Not selected"}
            </p>
            <p>
              <span className="text-gray-600">Session Formats:</span>{" "}
              {formData.sessionFormats.length || 0} selected
            </p>
            <p>
              <span className="text-gray-600">New Patients:</span>{" "}
              {formData.acceptingNewPatients ? "Accepting" : "Not accepting"}
            </p>
          </div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-900 mb-3">Education</h3>
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-gray-600">Degrees:</span>{" "}
              {formData.education.length || 0} entries
            </p>
            <p>
              <span className="text-gray-600">Certifications:</span>{" "}
              {formData.certifications.length || 0} listed
            </p>
          </div>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-900 mb-3">Contact Info</h3>
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-gray-600">Phone:</span>{" "}
              {formData.phone || "Not provided"}
            </p>
            <p>
              <span className="text-gray-600">Office Address:</span>{" "}
              {formData.officeAddress.city && formData.officeAddress.state
                ? `${formData.officeAddress.city}, ${formData.officeAddress.state}`
                : "Not provided"}
            </p>
          </div>
        </div>
      </div>

      {/* Final Note */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start">
          <div className="shrink-0">
            <CheckCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">
              Ready to complete your profile?
            </h3>
            <p className="text-sm text-yellow-700 mt-1">
              Once you submit, your profile will be reviewed for approval.
              You'll be able to start managing patients once approved.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
