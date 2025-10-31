'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Button } from '../../components/ui/button';
import { 
  User, 
  Users, 
  Heart, 
  Shield, 
  Phone, 
  MapPin, 
  Calendar,
  FileText,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';

export default function PatientDetailsPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [errors, setErrors] = useState({});
  const [formInitialized, setFormInitialized] = useState(false);
  const [saveStatus, setSaveStatus] = useState('');
  const [lastSaved, setLastSaved] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [completionPercentage, setCompletionPercentage] = useState(0);
  const [formData, setFormData] = useState({
    personalInfo: {
      dateOfBirth: '',
      gender: '',
      phoneNumber: '',
      address: {
        street: '',
        city: '',
        state: '',
        zipCode: '',
        country: 'United States'
      },
      occupation: '',
      maritalStatus: ''
    },
    guardianInfo: {
      isMinor: false,
      guardians: [{
        firstName: '',
        lastName: '',
        relationship: '',
        phoneNumber: '',
        email: '',
        address: {
          street: '',
          city: '',
          state: '',
          zipCode: '',
          country: 'United States'
        },
        isPrimary: true
      }]
    },
    medicalInfo: {
      primaryPhysician: {
        name: '',
        phone: '',
        email: ''
      },
      allergies: [''],
      currentMedications: [{
        name: '',
        dosage: '',
        frequency: '',
        prescribedBy: ''
      }],
      medicalConditions: [''],
      previousTherapy: {
        hasHadTherapy: false,
        therapistName: '',
        duration: '',
        reason: ''
      }
    },
    insuranceInfo: {
      hasInsurance: false,
      provider: '',
      policyNumber: '',
      groupNumber: '',
      subscriberName: '',
      subscriberRelationship: 'self'
    },
    preferences: {
      preferredContactMethod: 'email',
      therapistGenderPreference: 'no-preference',
      sessionFormat: 'both'
    },
    consents: {
      treatmentConsent: false,
      hipaaConsent: false,
      communicationConsent: false,
      guardianConsent: false
    }
  });

  // Form persistence key
  const FORM_STORAGE_KEY = `patient-form-${user?.id || 'temp'}`;

  // Load form data from localStorage on component mount
  useEffect(() => {
    if (isLoaded && user) {
      const savedData = localStorage.getItem(FORM_STORAGE_KEY);
      const savedStep = localStorage.getItem(`${FORM_STORAGE_KEY}-step`);
      let hasLoadedSavedData = false;
      
      if (savedData) {
        try {
          const parsedData = JSON.parse(savedData);
          setFormData(parsedData);
          hasLoadedSavedData = true;
          
          // Load last saved timestamp
          const savedTimestamp = localStorage.getItem(`${FORM_STORAGE_KEY}-timestamp`);
          if (savedTimestamp) {
            setLastSaved(new Date(savedTimestamp));
          }
        } catch (error) {
          console.error('Error parsing saved form data:', error);
        }
      }
      
      if (savedStep) {
        setCurrentStep(parseInt(savedStep, 10) || 1);
      }
      
      if (hasLoadedSavedData) {
        setSaveStatus('Previous draft loaded');
        setTimeout(() => setSaveStatus(''), 3000);
      }
      
      setFormInitialized(true);
    }
  }, [isLoaded, user, FORM_STORAGE_KEY]);

  // Debounced save to localStorage
  useEffect(() => {
    if (formInitialized && user) {
      setIsSaving(true);
      
      // Debounce the save operation to avoid too frequent updates
      const saveTimer = setTimeout(() => {
        const timestamp = new Date().toISOString();
        localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(formData));
        localStorage.setItem(`${FORM_STORAGE_KEY}-timestamp`, timestamp);
        
        setLastSaved(new Date());
        setIsSaving(false);
        setSaveStatus('Draft saved');
        
        // Clear the save status after 2 seconds
        setTimeout(() => {
          setSaveStatus('');
        }, 2000);
      }, 500); // 500ms debounce
      
      return () => {
        clearTimeout(saveTimer);
        setIsSaving(false);
      };
    }
  }, [formData, formInitialized, user, FORM_STORAGE_KEY]);

  // Save current step to localStorage whenever currentStep changes
  useEffect(() => {
    if (formInitialized && user) {
      localStorage.setItem(`${FORM_STORAGE_KEY}-step`, currentStep.toString());
    }
  }, [currentStep, formInitialized, user, FORM_STORAGE_KEY]);

  // Check if user is already a patient and redirect if needed
  useEffect(() => {
    if (isLoaded && user && user.unsafeMetadata?.role !== 'patient') {
      router.push('/dashboard');
    }
  }, [isLoaded, user, router]);

  // Calculate completion percentage
  const calculateCompletionPercentage = (data) => {
    let totalFields = 0;
    let filledFields = 0;

    // Personal info (required fields)
    const personalRequired = ['dateOfBirth', 'gender', 'phoneNumber'];
    const addressRequired = ['street', 'city', 'state', 'zipCode'];
    
    personalRequired.forEach(field => {
      totalFields++;
      if (data.personalInfo[field]) filledFields++;
    });
    
    addressRequired.forEach(field => {
      totalFields++;
      if (data.personalInfo.address[field]) filledFields++;
    });

    // Guardian info (if minor or if any guardian field is filled)
    if (data.guardianInfo.isMinor || data.guardianInfo.guardians[0].firstName) {
      const guardianRequired = ['firstName', 'lastName', 'relationship', 'phoneNumber'];
      guardianRequired.forEach(field => {
        totalFields++;
        if (data.guardianInfo.guardians[0][field]) filledFields++;
      });
    }

    // Consents (always required)
    const consentRequired = ['treatmentConsent', 'hipaaConsent', 'communicationConsent'];
    consentRequired.forEach(field => {
      totalFields++;
      if (data.consents[field]) filledFields++;
    });

    if (data.guardianInfo.isMinor) {
      totalFields++;
      if (data.consents.guardianConsent) filledFields++;
    }

    return Math.round((filledFields / totalFields) * 100);
  };

  // Calculate if user is a minor based on date of birth
  useEffect(() => {
    if (formData.personalInfo.dateOfBirth) {
      const birthDate = new Date(formData.personalInfo.dateOfBirth);
      const today = new Date();
      let age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
      }
      
      setFormData(prev => ({
        ...prev,
        guardianInfo: {
          ...prev.guardianInfo,
          isMinor: age < 18
        }
      }));
    }
  }, [formData.personalInfo.dateOfBirth]);

  // Update completion percentage when form data changes
  useEffect(() => {
    if (formInitialized) {
      const percentage = calculateCompletionPercentage(formData);
      setCompletionPercentage(percentage);
    }
  }, [formData, formInitialized]);

  const handleInputChange = (section, field, value, index = null, subField = null) => {
    setFormData(prev => {
      const newData = { ...prev };
      
      if (index !== null) {
        // Handle array fields
        if (subField) {
          newData[section][field][index][subField] = value;
        } else {
          newData[section][field][index] = value;
        }
      } else if (subField) {
        // Handle nested object fields
        newData[section][field][subField] = value;
      } else {
        // Handle regular fields
        newData[section][field] = value;
      }
      
      return newData;
    });
  };

  const addArrayItem = (section, field, defaultItem) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: [...prev[section][field], defaultItem]
      }
    }));
  };

  const removeArrayItem = (section, field, index) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: prev[section][field].filter((_, i) => i !== index)
      }
    }));
  };

  const validateStep = (step) => {
    const newErrors = {};
    
    switch (step) {
      case 1:
        if (!formData.personalInfo.dateOfBirth) newErrors.dateOfBirth = 'Date of birth is required';
        if (!formData.personalInfo.gender) newErrors.gender = 'Gender is required';
        if (!formData.personalInfo.phoneNumber) newErrors.phoneNumber = 'Phone number is required';
        if (!formData.personalInfo.address.street) newErrors.street = 'Street address is required';
        if (!formData.personalInfo.address.city) newErrors.city = 'City is required';
        if (!formData.personalInfo.address.state) newErrors.state = 'State is required';
        if (!formData.personalInfo.address.zipCode) newErrors.zipCode = 'ZIP code is required';
        break;
      case 2:
        if (formData.guardianInfo.isMinor || formData.guardianInfo.guardians[0].firstName) {
          const guardian = formData.guardianInfo.guardians[0];
          if (!guardian.firstName) newErrors.guardianFirstName = 'Guardian first name is required';
          if (!guardian.lastName) newErrors.guardianLastName = 'Guardian last name is required';
          if (!guardian.relationship) newErrors.guardianRelationship = 'Relationship is required';
          if (!guardian.phoneNumber) newErrors.guardianPhone = 'Guardian phone number is required';
        }
        break;
      case 4:
        if (!formData.consents.treatmentConsent) newErrors.treatmentConsent = 'Treatment consent is required';
        if (!formData.consents.hipaaConsent) newErrors.hipaaConsent = 'HIPAA consent is required';
        if (!formData.consents.communicationConsent) newErrors.communicationConsent = 'Communication consent is required';
        if (formData.guardianInfo.isMinor && !formData.consents.guardianConsent) {
          newErrors.guardianConsent = 'Guardian consent is required for minors';
        }
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, 4));
    }
  };

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const clearSavedFormData = (skipConfirm = false) => {
    if (skipConfirm || confirm('Are you sure you want to clear all saved form data? This action cannot be undone.')) {
      if (user) {
        localStorage.removeItem(FORM_STORAGE_KEY);
        localStorage.removeItem(`${FORM_STORAGE_KEY}-step`);
        localStorage.removeItem(`${FORM_STORAGE_KEY}-timestamp`);
        
        // Reset form to initial state
        setFormData({
          personalInfo: {
            dateOfBirth: '',
            gender: '',
            phoneNumber: '',
            address: {
              street: '',
              city: '',
              state: '',
              zipCode: '',
              country: 'United States'
            },
            occupation: '',
            maritalStatus: ''
          },
          guardianInfo: {
            isMinor: false,
            guardians: [{
              firstName: '',
              lastName: '',
              relationship: '',
              phoneNumber: '',
              email: '',
              address: {
                street: '',
                city: '',
                state: '',
                zipCode: '',
                country: 'United States'
              },
              isPrimary: true
            }]
          },
          medicalInfo: {
            primaryPhysician: {
              name: '',
              phone: '',
              email: ''
            },
            allergies: [''],
            currentMedications: [{
              name: '',
              dosage: '',
              frequency: '',
              prescribedBy: ''
            }],
            medicalConditions: [''],
            previousTherapy: {
              hasHadTherapy: false,
              therapistName: '',
              duration: '',
              reason: ''
            }
          },
          insuranceInfo: {
            hasInsurance: false,
            provider: '',
            policyNumber: '',
            groupNumber: '',
            subscriberName: '',
            subscriberRelationship: 'self'
          },
          preferences: {
            preferredContactMethod: 'email',
            therapistGenderPreference: 'no-preference',
            sessionFormat: 'both'
          },
          consents: {
            treatmentConsent: false,
            hipaaConsent: false,
            communicationConsent: false,
            guardianConsent: false
          }
        });
        
        setCurrentStep(1);
        setErrors({});
        setLastSaved(null);
        setSaveStatus('Form data cleared');
        
        setTimeout(() => setSaveStatus(''), 2000);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateStep(4)) return;
    
    setLoading(true);
    
    try {
      console.log('Submitting form data...');
      console.log('User:', user);
      console.log('User ID:', user?.id);
      console.log('User role:', user?.unsafeMetadata?.role);
      
      // First test a simple auth endpoint
      console.log('Testing basic auth first...');
      const authTest = await fetch('/api/debug-auth', { method: 'POST' });
      const authResult = await authTest.json();
      console.log('Auth test result:', authResult);
      
      if (!authTest.ok) {
        console.error('Auth test failed:', authResult);
        setErrors({ submit: `Authentication failed: ${authResult.error}` });
        return;
      }
      
      const response = await fetch('/api/patients', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          consents: {
            ...formData.consents,
            consentDate: new Date()
          }
        }),
      });

      if (response.ok) {
        // Clear saved form data on successful submission
        clearSavedFormData(true); // Skip confirmation dialog
        router.push('/dashboard');
      } else {
        const error = await response.json();
        console.error('API Response Status:', response.status);
        console.error('API Response Error:', error);
        setErrors({ submit: error.error || 'Failed to save patient details' });
      }
    } catch (error) {
      console.error('Error submitting form:', error);
      setErrors({ submit: 'An unexpected error occurred' });
    } finally {
      setLoading(false);
    }
  };

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Debug user authentication status
  console.log('Patient Details Page - User Auth Status:');
  console.log('isLoaded:', isLoaded);
  console.log('user:', user);
  console.log('user.id:', user?.id);
  console.log('user.emailAddresses:', user?.emailAddresses);
  console.log('user.unsafeMetadata:', user?.unsafeMetadata);

  // If user is not loaded or not authenticated, redirect to sign-in
  if (!user) {
    console.log('No user found, redirecting to sign-in');
    router.push('/sign-in');
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-4" />
          <p className="text-gray-600">Redirecting to sign in...</p>
        </div>
      </div>
    );
  }

  const steps = [
    { number: 1, title: 'Personal Information', icon: User },
    { number: 2, title: 'Guardian/Emergency Contact', icon: Users },
    { number: 3, title: 'Medical & Insurance', icon: Heart },
    { number: 4, title: 'Consent & Preferences', icon: Shield }
  ];

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Complete Your Patient Profile</h1>
          <p className="text-gray-600">Help us provide you with the best care by completing your profile</p>
          
          {/* Save Status and Actions */}
          <div className="flex items-center justify-center mt-4 space-x-4">
            {(saveStatus || isSaving) && (
              <div className="flex items-center text-sm">
                {isSaving ? (
                  <div className="flex items-center text-blue-600">
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                    Saving draft...
                  </div>
                ) : (
                  <div className="flex items-center text-green-600">
                    <CheckCircle className="w-4 h-4 mr-1" />
                    {saveStatus}
                  </div>
                )}
              </div>
            )}
            
            {formInitialized && (
              <div className="flex flex-col items-center space-y-1">
                <div className="flex items-center space-x-4">
                  {completionPercentage > 0 && (
                    <div className="text-xs text-indigo-600 font-medium">
                      {completionPercentage}% Complete
                    </div>
                  )}
                  {lastSaved && (
                    <div className="text-xs text-gray-400">
                      Last saved: {lastSaved.toLocaleTimeString()}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={clearSavedFormData}
                  className="text-xs text-gray-500 hover:text-red-600 underline"
                >
                  Clear saved data
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors duration-200 ${
                  currentStep >= step.number 
                    ? 'bg-indigo-600 border-indigo-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-400'
                }`}>
                  {currentStep > step.number ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <step.icon className="w-5 h-5" />
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-full h-1 mx-4 transition-colors duration-200 ${
                    currentStep > step.number 
                      ? 'bg-indigo-600' 
                      : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between text-sm text-gray-600">
            {steps.map(step => (
              <span key={step.number} className={`font-medium ${
                currentStep >= step.number ? 'text-indigo-600' : ''
              }`}>
                {step.title}
              </span>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <Card className="shadow-xl border-0">
            <CardHeader className="bg-linear-to-r from-indigo-500 to-purple-600 text-white rounded-t-lg">
              <CardTitle className="text-xl">
                Step {currentStep}: {steps[currentStep - 1].title}
              </CardTitle>
              <CardDescription className="text-indigo-100">
                {currentStep === 1 && 'Please provide your basic personal information'}
                {currentStep === 2 && 'Emergency contact and guardian information'}
                {currentStep === 3 && 'Medical history and insurance details'}
                {currentStep === 4 && 'Consent forms and preferences'}
              </CardDescription>
            </CardHeader>

            <CardContent className="p-8">
              {/* Step 1: Personal Information */}
              {currentStep === 1 && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <Label htmlFor="dateOfBirth">Date of Birth *</Label>
                      <Input
                        id="dateOfBirth"
                        type="date"
                        value={formData.personalInfo.dateOfBirth}
                        onChange={(e) => handleInputChange('personalInfo', 'dateOfBirth', e.target.value)}
                        className={errors.dateOfBirth ? 'border-red-500' : ''}
                      />
                      {errors.dateOfBirth && <p className="text-red-500 text-sm mt-1">{errors.dateOfBirth}</p>}
                    </div>

                    <div>
                      <Label htmlFor="gender">Gender *</Label>
                      <select
                        id="gender"
                        value={formData.personalInfo.gender}
                        onChange={(e) => handleInputChange('personalInfo', 'gender', e.target.value)}
                        className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${errors.gender ? 'border-red-500' : ''}`}
                      >
                        <option value="">Select Gender</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                        <option value="prefer-not-to-say">Prefer not to say</option>
                      </select>
                      {errors.gender && <p className="text-red-500 text-sm mt-1">{errors.gender}</p>}
                    </div>

                    <div>
                      <Label htmlFor="phoneNumber">Phone Number *</Label>
                      <Input
                        id="phoneNumber"
                        type="tel"
                        placeholder="(555) 123-4567"
                        value={formData.personalInfo.phoneNumber}
                        onChange={(e) => handleInputChange('personalInfo', 'phoneNumber', e.target.value)}
                        className={errors.phoneNumber ? 'border-red-500' : ''}
                      />
                      {errors.phoneNumber && <p className="text-red-500 text-sm mt-1">{errors.phoneNumber}</p>}
                    </div>

                    <div>
                      <Label htmlFor="occupation">Occupation</Label>
                      <Input
                        id="occupation"
                        placeholder="Your occupation"
                        value={formData.personalInfo.occupation}
                        onChange={(e) => handleInputChange('personalInfo', 'occupation', e.target.value)}
                      />
                    </div>

                    <div>
                      <Label htmlFor="maritalStatus">Marital Status</Label>
                      <select
                        id="maritalStatus"
                        value={formData.personalInfo.maritalStatus}
                        onChange={(e) => handleInputChange('personalInfo', 'maritalStatus', e.target.value)}
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      >
                        <option value="">Select Status</option>
                        <option value="single">Single</option>
                        <option value="married">Married</option>
                        <option value="divorced">Divorced</option>
                        <option value="widowed">Widowed</option>
                        <option value="separated">Separated</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                  </div>

                  <div className="border-t pt-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <MapPin className="w-5 h-5 mr-2 text-indigo-600" />
                      Address Information
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="md:col-span-2">
                        <Label htmlFor="street">Street Address *</Label>
                        <Input
                          id="street"
                          placeholder="123 Main Street"
                          value={formData.personalInfo.address.street}
                          onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value, null, 'street')}
                          className={errors.street ? 'border-red-500' : ''}
                        />
                        {errors.street && <p className="text-red-500 text-sm mt-1">{errors.street}</p>}
                      </div>

                      <div>
                        <Label htmlFor="city">City *</Label>
                        <Input
                          id="city"
                          placeholder="City"
                          value={formData.personalInfo.address.city}
                          onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value, null, 'city')}
                          className={errors.city ? 'border-red-500' : ''}
                        />
                        {errors.city && <p className="text-red-500 text-sm mt-1">{errors.city}</p>}
                      </div>

                      <div>
                        <Label htmlFor="state">State *</Label>
                        <Input
                          id="state"
                          placeholder="State"
                          value={formData.personalInfo.address.state}
                          onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value, null, 'state')}
                          className={errors.state ? 'border-red-500' : ''}
                        />
                        {errors.state && <p className="text-red-500 text-sm mt-1">{errors.state}</p>}
                      </div>

                      <div>
                        <Label htmlFor="zipCode">ZIP Code *</Label>
                        <Input
                          id="zipCode"
                          placeholder="12345"
                          value={formData.personalInfo.address.zipCode}
                          onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value, null, 'zipCode')}
                          className={errors.zipCode ? 'border-red-500' : ''}
                        />
                        {errors.zipCode && <p className="text-red-500 text-sm mt-1">{errors.zipCode}</p>}
                      </div>

                      <div>
                        <Label htmlFor="country">Country</Label>
                        <Input
                          id="country"
                          value={formData.personalInfo.address.country}
                          onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value, null, 'country')}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2: Guardian/Emergency Contact */}
              {currentStep === 2 && (
                <div className="space-y-6">
                  {formData.guardianInfo.isMinor && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
                      <div className="flex items-center">
                        <AlertCircle className="w-5 h-5 text-yellow-600 mr-2" />
                        <p className="text-yellow-800 font-medium">
                          As you are under 18, guardian information is required.
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <Users className="w-5 h-5 mr-2 text-indigo-600" />
                      {formData.guardianInfo.isMinor ? 'Guardian Information' : 'Emergency Contact'}
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="guardianFirstName">
                          First Name {(formData.guardianInfo.isMinor || formData.guardianInfo.guardians[0].firstName) && '*'}
                        </Label>
                        <Input
                          id="guardianFirstName"
                          placeholder="First Name"
                          value={formData.guardianInfo.guardians[0].firstName}
                          onChange={(e) => handleInputChange('guardianInfo', 'guardians', e.target.value, 0, 'firstName')}
                          className={errors.guardianFirstName ? 'border-red-500' : ''}
                        />
                        {errors.guardianFirstName && <p className="text-red-500 text-sm mt-1">{errors.guardianFirstName}</p>}
                      </div>

                      <div>
                        <Label htmlFor="guardianLastName">
                          Last Name {(formData.guardianInfo.isMinor || formData.guardianInfo.guardians[0].lastName) && '*'}
                        </Label>
                        <Input
                          id="guardianLastName"
                          placeholder="Last Name"
                          value={formData.guardianInfo.guardians[0].lastName}
                          onChange={(e) => handleInputChange('guardianInfo', 'guardians', e.target.value, 0, 'lastName')}
                          className={errors.guardianLastName ? 'border-red-500' : ''}
                        />
                        {errors.guardianLastName && <p className="text-red-500 text-sm mt-1">{errors.guardianLastName}</p>}
                      </div>

                      <div>
                        <Label htmlFor="guardianRelationship">
                          Relationship {(formData.guardianInfo.isMinor || formData.guardianInfo.guardians[0].relationship) && '*'}
                        </Label>
                        <select
                          id="guardianRelationship"
                          value={formData.guardianInfo.guardians[0].relationship}
                          onChange={(e) => handleInputChange('guardianInfo', 'guardians', e.target.value, 0, 'relationship')}
                          className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${errors.guardianRelationship ? 'border-red-500' : ''}`}
                        >
                          <option value="">Select Relationship</option>
                          <option value="mother">Mother</option>
                          <option value="father">Father</option>
                          <option value="guardian">Guardian</option>
                          <option value="spouse">Spouse</option>
                          <option value="sibling">Sibling</option>
                          <option value="friend">Friend</option>
                          <option value="other">Other</option>
                        </select>
                        {errors.guardianRelationship && <p className="text-red-500 text-sm mt-1">{errors.guardianRelationship}</p>}
                      </div>

                      <div>
                        <Label htmlFor="guardianPhone">
                          Phone Number {(formData.guardianInfo.isMinor || formData.guardianInfo.guardians[0].phoneNumber) && '*'}
                        </Label>
                        <Input
                          id="guardianPhone"
                          type="tel"
                          placeholder="(555) 123-4567"
                          value={formData.guardianInfo.guardians[0].phoneNumber}
                          onChange={(e) => handleInputChange('guardianInfo', 'guardians', e.target.value, 0, 'phoneNumber')}
                          className={errors.guardianPhone ? 'border-red-500' : ''}
                        />
                        {errors.guardianPhone && <p className="text-red-500 text-sm mt-1">{errors.guardianPhone}</p>}
                      </div>

                      <div>
                        <Label htmlFor="guardianEmail">Email</Label>
                        <Input
                          id="guardianEmail"
                          type="email"
                          placeholder="guardian@example.com"
                          value={formData.guardianInfo.guardians[0].email}
                          onChange={(e) => handleInputChange('guardianInfo', 'guardians', e.target.value, 0, 'email')}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Step 3: Medical & Insurance */}
              {currentStep === 3 && (
                <div className="space-y-6">
                  {/* Primary Physician */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <Heart className="w-5 h-5 mr-2 text-indigo-600" />
                      Primary Physician
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <Label htmlFor="physicianName">Doctor's Name</Label>
                        <Input
                          id="physicianName"
                          placeholder="Dr. Smith"
                          value={formData.medicalInfo.primaryPhysician.name}
                          onChange={(e) => handleInputChange('medicalInfo', 'primaryPhysician', e.target.value, null, 'name')}
                        />
                      </div>
                      <div>
                        <Label htmlFor="physicianPhone">Phone</Label>
                        <Input
                          id="physicianPhone"
                          type="tel"
                          placeholder="(555) 123-4567"
                          value={formData.medicalInfo.primaryPhysician.phone}
                          onChange={(e) => handleInputChange('medicalInfo', 'primaryPhysician', e.target.value, null, 'phone')}
                        />
                      </div>
                      <div>
                        <Label htmlFor="physicianEmail">Email</Label>
                        <Input
                          id="physicianEmail"
                          type="email"
                          placeholder="doctor@clinic.com"
                          value={formData.medicalInfo.primaryPhysician.email}
                          onChange={(e) => handleInputChange('medicalInfo', 'primaryPhysician', e.target.value, null, 'email')}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Allergies */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4">Allergies</h3>
                    {formData.medicalInfo.allergies.map((allergy, index) => (
                      <div key={index} className="flex items-center gap-2 mb-2">
                        <Input
                          placeholder="Allergy (e.g., Penicillin, Peanuts)"
                          value={allergy}
                          onChange={(e) => handleInputChange('medicalInfo', 'allergies', e.target.value, index)}
                        />
                        {formData.medicalInfo.allergies.length > 1 && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => removeArrayItem('medicalInfo', 'allergies', index)}
                          >
                            Remove
                          </Button>
                        )}
                      </div>
                    ))}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => addArrayItem('medicalInfo', 'allergies', '')}
                      className="mt-2"
                    >
                      Add Allergy
                    </Button>
                  </div>

                  {/* Previous Therapy */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4">Previous Therapy Experience</h3>
                    <div className="mb-4">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={formData.medicalInfo.previousTherapy.hasHadTherapy}
                          onChange={(e) => handleInputChange('medicalInfo', 'previousTherapy', e.target.checked, null, 'hasHadTherapy')}
                          className="mr-2"
                        />
                        I have had therapy before
                      </label>
                    </div>
                    
                    {formData.medicalInfo.previousTherapy.hasHadTherapy && (
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="therapistName">Previous Therapist Name</Label>
                          <Input
                            id="therapistName"
                            placeholder="Therapist name"
                            value={formData.medicalInfo.previousTherapy.therapistName}
                            onChange={(e) => handleInputChange('medicalInfo', 'previousTherapy', e.target.value, null, 'therapistName')}
                          />
                        </div>
                        <div>
                          <Label htmlFor="therapyDuration">Duration</Label>
                          <Input
                            id="therapyDuration"
                            placeholder="e.g., 6 months, 2 years"
                            value={formData.medicalInfo.previousTherapy.duration}
                            onChange={(e) => handleInputChange('medicalInfo', 'previousTherapy', e.target.value, null, 'duration')}
                          />
                        </div>
                        <div>
                          <Label htmlFor="therapyReason">Reason for therapy</Label>
                          <Input
                            id="therapyReason"
                            placeholder="Brief description"
                            value={formData.medicalInfo.previousTherapy.reason}
                            onChange={(e) => handleInputChange('medicalInfo', 'previousTherapy', e.target.value, null, 'reason')}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Insurance */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4">Insurance Information</h3>
                    <div className="mb-4">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={formData.insuranceInfo.hasInsurance}
                          onChange={(e) => handleInputChange('insuranceInfo', 'hasInsurance', e.target.checked)}
                          className="mr-2"
                        />
                        I have health insurance
                      </label>
                    </div>
                    
                    {formData.insuranceInfo.hasInsurance && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="insuranceProvider">Insurance Provider</Label>
                          <Input
                            id="insuranceProvider"
                            placeholder="Blue Cross, Aetna, etc."
                            value={formData.insuranceInfo.provider}
                            onChange={(e) => handleInputChange('insuranceInfo', 'provider', e.target.value)}
                          />
                        </div>
                        <div>
                          <Label htmlFor="policyNumber">Policy Number</Label>
                          <Input
                            id="policyNumber"
                            placeholder="Policy number"
                            value={formData.insuranceInfo.policyNumber}
                            onChange={(e) => handleInputChange('insuranceInfo', 'policyNumber', e.target.value)}
                          />
                        </div>
                        <div>
                          <Label htmlFor="groupNumber">Group Number</Label>
                          <Input
                            id="groupNumber"
                            placeholder="Group number"
                            value={formData.insuranceInfo.groupNumber}
                            onChange={(e) => handleInputChange('insuranceInfo', 'groupNumber', e.target.value)}
                          />
                        </div>
                        <div>
                          <Label htmlFor="subscriberName">Subscriber Name</Label>
                          <Input
                            id="subscriberName"
                            placeholder="Policy holder name"
                            value={formData.insuranceInfo.subscriberName}
                            onChange={(e) => handleInputChange('insuranceInfo', 'subscriberName', e.target.value)}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Step 4: Consent & Preferences */}
              {currentStep === 4 && (
                <div className="space-y-6">
                  {/* Preferences */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4">Preferences</h3>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="contactMethod">Preferred Contact Method</Label>
                        <select
                          id="contactMethod"
                          value={formData.preferences.preferredContactMethod}
                          onChange={(e) => handleInputChange('preferences', 'preferredContactMethod', e.target.value)}
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        >
                          <option value="email">Email</option>
                          <option value="phone">Phone</option>
                          <option value="text">Text Message</option>
                        </select>
                      </div>
                      
                      <div>
                        <Label htmlFor="therapistGender">Therapist Gender Preference</Label>
                        <select
                          id="therapistGender"
                          value={formData.preferences.therapistGenderPreference}
                          onChange={(e) => handleInputChange('preferences', 'therapistGenderPreference', e.target.value)}
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        >
                          <option value="no-preference">No Preference</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                        </select>
                      </div>
                      
                      <div>
                        <Label htmlFor="sessionFormat">Session Format Preference</Label>
                        <select
                          id="sessionFormat"
                          value={formData.preferences.sessionFormat}
                          onChange={(e) => handleInputChange('preferences', 'sessionFormat', e.target.value)}
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        >
                          <option value="both">Both In-Person and Virtual</option>
                          <option value="in-person">In-Person Only</option>
                          <option value="virtual">Virtual Only</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Consent Forms */}
                  <div className="border rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <Shield className="w-5 h-5 mr-2 text-indigo-600" />
                      Required Consents
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <label className="flex items-start">
                          <input
                            type="checkbox"
                            checked={formData.consents.treatmentConsent}
                            onChange={(e) => handleInputChange('consents', 'treatmentConsent', e.target.checked)}
                            className={`mr-3 mt-1 ${errors.treatmentConsent ? 'border-red-500' : ''}`}
                          />
                          <div>
                            <span className="font-medium">Treatment Consent *</span>
                            <p className="text-sm text-gray-600 mt-1">
                              I consent to receive mental health treatment and understand the nature of the services provided.
                            </p>
                          </div>
                        </label>
                        {errors.treatmentConsent && <p className="text-red-500 text-sm mt-1 ml-6">{errors.treatmentConsent}</p>}
                      </div>

                      <div>
                        <label className="flex items-start">
                          <input
                            type="checkbox"
                            checked={formData.consents.hipaaConsent}
                            onChange={(e) => handleInputChange('consents', 'hipaaConsent', e.target.checked)}
                            className={`mr-3 mt-1 ${errors.hipaaConsent ? 'border-red-500' : ''}`}
                          />
                          <div>
                            <span className="font-medium">HIPAA Consent *</span>
                            <p className="text-sm text-gray-600 mt-1">
                              I acknowledge that I have received and understand the HIPAA Privacy Notice.
                            </p>
                          </div>
                        </label>
                        {errors.hipaaConsent && <p className="text-red-500 text-sm mt-1 ml-6">{errors.hipaaConsent}</p>}
                      </div>

                      <div>
                        <label className="flex items-start">
                          <input
                            type="checkbox"
                            checked={formData.consents.communicationConsent}
                            onChange={(e) => handleInputChange('consents', 'communicationConsent', e.target.checked)}
                            className={`mr-3 mt-1 ${errors.communicationConsent ? 'border-red-500' : ''}`}
                          />
                          <div>
                            <span className="font-medium">Communication Consent *</span>
                            <p className="text-sm text-gray-600 mt-1">
                              I consent to receive communications regarding appointments, treatment, and billing.
                            </p>
                          </div>
                        </label>
                        {errors.communicationConsent && <p className="text-red-500 text-sm mt-1 ml-6">{errors.communicationConsent}</p>}
                      </div>

                      {formData.guardianInfo.isMinor && (
                        <div>
                          <label className="flex items-start">
                            <input
                              type="checkbox"
                              checked={formData.consents.guardianConsent}
                              onChange={(e) => handleInputChange('consents', 'guardianConsent', e.target.checked)}
                              className={`mr-3 mt-1 ${errors.guardianConsent ? 'border-red-500' : ''}`}
                            />
                            <div>
                              <span className="font-medium">Guardian Consent *</span>
                              <p className="text-sm text-gray-600 mt-1">
                                As the guardian, I consent to mental health treatment for the minor patient.
                              </p>
                            </div>
                          </label>
                          {errors.guardianConsent && <p className="text-red-500 text-sm mt-1 ml-6">{errors.guardianConsent}</p>}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {errors.submit && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-center">
                    <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
                    <p className="text-red-800">{errors.submit}</p>
                  </div>
                </div>
              )}

              {/* Debug Auth Button - Remove in production */}
              {/* {process.env.NODE_ENV === 'development' && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-blue-800 text-sm mb-2">Debug: Authentication Status</p>
                  <div className="text-xs text-gray-600 mb-2">
                    <p>User ID: {user?.id || 'Not found'}</p>
                    <p>Email: {user?.emailAddresses?.[0]?.emailAddress || 'Not found'}</p>
                    <p>Role: {user?.unsafeMetadata?.role || 'Not set'}</p>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          console.log('Testing API auth...');
                          const response = await fetch('/api/test-auth');
                          const data = await response.json();
                          console.log('Auth test result:', data);
                          alert(`Auth test: ${response.ok ? 'SUCCESS' : 'FAILED'}\n${JSON.stringify(data, null, 2)}`);
                        } catch (error) {
                          console.error('Auth test failed:', error);
                          alert(`Auth test failed: ${error.message}`);
                        }
                      }}
                      className="bg-blue-600 text-white py-1 px-3 rounded text-xs"
                    >
                      Test API Auth
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        console.log('Current user object:', user);
                        console.log('Document cookies:', document.cookie);
                        alert(`Frontend Auth:\nUser: ${user ? 'Found' : 'Not found'}\nID: ${user?.id}\nEmail: ${user?.emailAddresses?.[0]?.emailAddress}`);
                      }}
                      className="bg-green-600 text-white py-1 px-3 rounded text-xs"
                    >
                      Check Frontend Auth
                    </button>
                  </div>
                </div>
              )} */}
            </CardContent>

            {/* Navigation Buttons */}
            <div className="flex justify-between items-center p-6 bg-gray-50 rounded-b-lg">
              <Button
                type="button"
                variant="outline"
                onClick={prevStep}
                disabled={currentStep === 1}
              >
                Previous
              </Button>

              <span className="text-sm text-gray-600">
                Step {currentStep} of {steps.length}
              </span>

              {currentStep < steps.length ? (
                <Button className={"bg-green-500"} type="button" onClick={nextStep}>
                  Next
                </Button>
              ) : (
                <Button className={"bg-green-500"} type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Complete Profile'
                  )}
                </Button>
              )}
            </div>
          </Card>
        </form>
      </div>
    </div>
  );
}
