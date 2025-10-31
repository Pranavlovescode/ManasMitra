'use client';

import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Loader2, Users, UserCheck, Search, Calendar, Phone, Mail, MapPin, X, Activity, Heart, AlertCircle, Pill } from 'lucide-react';

export default function ManagePatientsPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [patients, setPatients] = useState({
    allPatients: [],
    assignedPatients: [],
    unassignedPatients: [],
    stats: {}
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);

  useEffect(() => {
    if (isLoaded && user) {
      if (user.unsafeMetadata?.role !== 'therapist') {
        router.push('/dashboard');
        return;
      }
      fetchPatients();
    }
  }, [isLoaded, user, router]);

  const fetchPatients = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching patients from API...');
      const response = await fetch('/api/therapists/patients');
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('API Error:', errorData);
        throw new Error(errorData.message || errorData.error || 'Failed to fetch patients');
      }

      const data = await response.json();
      console.log('Patients data received:', data);
      
      setPatients(data);
    } catch (err) {
      console.error('Error fetching patients:', err);
      setError(err.message || 'An error occurred while fetching patients');
    } finally {
      setLoading(false);
    }
  };

  const filterPatients = (patientList) => {
    if (!searchTerm) return patientList;
    
    return patientList.filter(patient => {
      const fullName = `${patient.userId?.firstName || ''} ${patient.userId?.lastName || ''}`.toLowerCase();
      const email = patient.userId?.email?.toLowerCase() || '';
      return fullName.includes(searchTerm.toLowerCase()) || email.includes(searchTerm.toLowerCase());
    });
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const handlePatientClick = (patient) => {
    setSelectedPatient(patient);
    setShowDetailsModal(true);
  };

  const closeDetailsModal = () => {
    setShowDetailsModal(false);
    setSelectedPatient(null);
  };

  const PatientCard = ({ patient, showFullDetails = false }) => (
    <Card 
      className="hover:shadow-lg transition-shadow duration-200 cursor-pointer"
      onClick={() => handlePatientClick(patient)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-linear-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold text-lg">
              {patient.userId?.firstName?.[0]}{patient.userId?.lastName?.[0]}
            </div>
            <div>
              <CardTitle className="text-lg">
                {patient.userId?.firstName} {patient.userId?.lastName}
              </CardTitle>
              <CardDescription className="flex items-center gap-1 text-sm">
                <Mail className="w-3 h-3" />
                {patient.userId?.email}
              </CardDescription>
            </div>
          </div>
          {patient.isAssigned && (
            <span className="px-3 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded-full">
              Assigned
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500 flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                Age
              </p>
              <p className="font-medium">{patient.age || 'N/A'} years</p>
            </div>
            <div>
              <p className="text-gray-500">Gender</p>
              <p className="font-medium capitalize">{patient.personalInfo?.gender || 'N/A'}</p>
            </div>
            {patient.personalInfo?.phoneNumber && (
              <div className="col-span-2">
                <p className="text-gray-500 flex items-center gap-1">
                  <Phone className="w-4 h-4" />
                  Phone
                </p>
                <p className="font-medium">{patient.personalInfo.phoneNumber}</p>
              </div>
            )}
            {patient.personalInfo?.address?.city && (
              <div className="col-span-2">
                <p className="text-gray-500 flex items-center gap-1">
                  <MapPin className="w-4 h-4" />
                  Location
                </p>
                <p className="font-medium">
                  {patient.personalInfo.address.city}, {patient.personalInfo.address.state}
                </p>
              </div>
            )}
          </div>

          {showFullDetails && (
            <div className="mt-4 pt-4 border-t space-y-3">
              {patient.medicalInfo?.allergies?.length > 0 && (
                <div>
                  <p className="text-gray-500 text-sm">Allergies</p>
                  <p className="font-medium text-sm">
                    {patient.medicalInfo.allergies.filter(a => a).join(', ') || 'None'}
                  </p>
                </div>
              )}
              {patient.preferences?.sessionFormat && (
                <div>
                  <p className="text-gray-500 text-sm">Session Preference</p>
                  <p className="font-medium text-sm capitalize">
                    {patient.preferences.sessionFormat}
                  </p>
                </div>
              )}
              {patient.preferences?.preferredContactMethod && (
                <div>
                  <p className="text-gray-500 text-sm">Preferred Contact</p>
                  <p className="font-medium text-sm capitalize">
                    {patient.preferences.preferredContactMethod}
                  </p>
                </div>
              )}
            </div>
          )}

          <div className="mt-3 pt-3 border-t text-xs text-gray-500">
            Registered: {formatDate(patient.createdAt)}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  if (!isLoaded || loading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading patients...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-red-600">Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  const filteredAssignedPatients = filterPatients(patients.assignedPatients);
  const filteredUnassignedPatients = filterPatients(patients.unassignedPatients);

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="backdrop-blur-md bg-white/80 border-b border-white/20 shadow-sm sticky top-0 z-10">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Manage Patients</h1>
              <p className="text-gray-600 mt-1">View and manage your patient assignments</p>
            </div>
            <button
              onClick={() => router.push('/therapist/dashboard')}
              className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              ← Back to Dashboard
            </button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-white/80 backdrop-blur-sm shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Total Patients</p>
                  <p className="text-3xl font-bold text-gray-800">{patients.stats.totalPatients || 0}</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                  <Users className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 backdrop-blur-sm shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Assigned to You</p>
                  <p className="text-3xl font-bold text-green-600">{patients.stats.assignedCount || 0}</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                  <UserCheck className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 backdrop-blur-sm shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Other Patients</p>
                  <p className="text-3xl font-bold text-gray-600">{patients.stats.unassignedCount || 0}</p>
                </div>
                <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
                  <Users className="w-6 h-6 text-gray-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Search Bar */}
        <Card className="mb-6 bg-white/80 backdrop-blur-sm shadow-lg">
          <CardContent className="p-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                type="text"
                placeholder="Search patients by name or email..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </CardContent>
        </Card>

        {/* Tabs */}
        <Card className="bg-white/80 backdrop-blur-sm shadow-xl">
          <CardContent className="p-6">
            <Tabs defaultValue="assigned" className="space-y-6">
              <TabsList className="grid w-full grid-cols-2 bg-gray-100/80 p-1 rounded-xl">
                <TabsTrigger 
                  value="assigned"
                  className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  <UserCheck className="w-4 h-4 mr-2" />
                  Assigned Patients ({filteredAssignedPatients.length})
                </TabsTrigger>
                <TabsTrigger 
                  value="all"
                  className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  <Users className="w-4 h-4 mr-2" />
                  All Patients ({filteredUnassignedPatients.length})
                </TabsTrigger>
              </TabsList>

              <TabsContent value="assigned" className="space-y-4 mt-6">
                {filteredAssignedPatients.length === 0 ? (
                  <div className="text-center py-12">
                    <UserCheck className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">No Assigned Patients</h3>
                    <p className="text-gray-500">
                      {searchTerm ? 'No patients match your search criteria.' : 'You don\'t have any assigned patients yet.'}
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredAssignedPatients.map(patient => (
                      <PatientCard key={patient._id} patient={patient} showFullDetails={true} />
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="all" className="space-y-4 mt-6">
                {filteredUnassignedPatients.length === 0 ? (
                  <div className="text-center py-12">
                    <Users className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">No Other Patients</h3>
                    <p className="text-gray-500">
                      {searchTerm ? 'No patients match your search criteria.' : 'All patients are currently assigned.'}
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredUnassignedPatients.map(patient => (
                      <PatientCard key={patient._id} patient={patient} showFullDetails={false} />
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Patient Details Modal */}
      {showDetailsModal && selectedPatient && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="sticky top-0 bg-linear-to-r from-indigo-600 to-purple-600 text-white p-6 rounded-t-xl flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center text-2xl font-bold">
                  {selectedPatient.userId?.firstName?.[0]}{selectedPatient.userId?.lastName?.[0]}
                </div>
                <div>
                  <h2 className="text-2xl font-bold">
                    {selectedPatient.userId?.firstName} {selectedPatient.userId?.lastName}
                  </h2>
                  <p className="text-indigo-100">{selectedPatient.userId?.email}</p>
                </div>
              </div>
              <button
                onClick={closeDetailsModal}
                className="p-2 hover:bg-white/20 rounded-full transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-6">
              {/* Personal Information */}
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Users className="w-5 h-5 text-indigo-600" />
                  Personal Information
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 bg-gray-50 p-4 rounded-lg">
                  <div>
                    <p className="text-sm text-gray-500">Age</p>
                    <p className="font-medium">{selectedPatient.age || 'N/A'} years</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Gender</p>
                    <p className="font-medium capitalize">{selectedPatient.personalInfo?.gender || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Marital Status</p>
                    <p className="font-medium capitalize">{selectedPatient.personalInfo?.maritalStatus || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Occupation</p>
                    <p className="font-medium">{selectedPatient.personalInfo?.occupation || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Phone</p>
                    <p className="font-medium">{selectedPatient.personalInfo?.phoneNumber || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Date of Birth</p>
                    <p className="font-medium">{formatDate(selectedPatient.personalInfo?.dateOfBirth)}</p>
                  </div>
                </div>
              </div>

              {/* Address */}
              {selectedPatient.personalInfo?.address && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <MapPin className="w-5 h-5 text-indigo-600" />
                    Address
                  </h3>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="font-medium">
                      {selectedPatient.personalInfo.address.street && `${selectedPatient.personalInfo.address.street}, `}
                      {selectedPatient.personalInfo.address.city && `${selectedPatient.personalInfo.address.city}, `}
                      {selectedPatient.personalInfo.address.state && `${selectedPatient.personalInfo.address.state} `}
                      {selectedPatient.personalInfo.address.zipCode}
                    </p>
                    {selectedPatient.personalInfo.address.country && (
                      <p className="text-gray-600 mt-1">{selectedPatient.personalInfo.address.country}</p>
                    )}
                  </div>
                </div>
              )}

              {/* Medical Information */}
              {selectedPatient.medicalInfo && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <Heart className="w-5 h-5 text-red-600" />
                    Medical Information
                  </h3>
                  <div className="space-y-4">
                    {/* Allergies */}
                    {selectedPatient.medicalInfo.allergies?.length > 0 && (
                      <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                        <p className="text-sm font-semibold text-red-800 mb-2 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" />
                          Allergies
                        </p>
                        <p className="text-red-700">
                          {selectedPatient.medicalInfo.allergies.filter(a => a).join(', ') || 'None reported'}
                        </p>
                      </div>
                    )}

                    {/* Current Medications */}
                    {selectedPatient.medicalInfo.currentMedications?.length > 0 && 
                     selectedPatient.medicalInfo.currentMedications.some(med => med.name) && (
                      <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                        <p className="text-sm font-semibold text-blue-800 mb-3 flex items-center gap-2">
                          <Pill className="w-4 h-4" />
                          Current Medications
                        </p>
                        <div className="space-y-2">
                          {selectedPatient.medicalInfo.currentMedications
                            .filter(med => med.name)
                            .map((med, index) => (
                              <div key={index} className="bg-white p-3 rounded">
                                <p className="font-medium text-blue-900">{med.name}</p>
                                {med.dosage && <p className="text-sm text-blue-700">Dosage: {med.dosage}</p>}
                                {med.frequency && <p className="text-sm text-blue-700">Frequency: {med.frequency}</p>}
                                {med.prescribedBy && <p className="text-sm text-blue-600">Prescribed by: {med.prescribedBy}</p>}
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {/* Medical Conditions */}
                    {selectedPatient.medicalInfo.medicalConditions?.length > 0 && (
                      <div className="bg-orange-50 border border-orange-200 p-4 rounded-lg">
                        <p className="text-sm font-semibold text-orange-800 mb-2">Medical Conditions</p>
                        <p className="text-orange-700">
                          {selectedPatient.medicalInfo.medicalConditions.filter(c => c).join(', ') || 'None reported'}
                        </p>
                      </div>
                    )}

                    {/* Primary Physician */}
                    {selectedPatient.medicalInfo.primaryPhysician?.name && (
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <p className="text-sm font-semibold text-gray-800 mb-2">Primary Physician</p>
                        <p className="font-medium">{selectedPatient.medicalInfo.primaryPhysician.name}</p>
                        {selectedPatient.medicalInfo.primaryPhysician.phone && (
                          <p className="text-sm text-gray-600">Phone: {selectedPatient.medicalInfo.primaryPhysician.phone}</p>
                        )}
                        {selectedPatient.medicalInfo.primaryPhysician.email && (
                          <p className="text-sm text-gray-600">Email: {selectedPatient.medicalInfo.primaryPhysician.email}</p>
                        )}
                      </div>
                    )}

                    {/* Previous Therapy */}
                    {selectedPatient.medicalInfo.previousTherapy?.hasHadTherapy && (
                      <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg">
                        <p className="text-sm font-semibold text-purple-800 mb-2">Previous Therapy Experience</p>
                        {selectedPatient.medicalInfo.previousTherapy.therapistName && (
                          <p className="text-purple-700">Therapist: {selectedPatient.medicalInfo.previousTherapy.therapistName}</p>
                        )}
                        {selectedPatient.medicalInfo.previousTherapy.duration && (
                          <p className="text-purple-700">Duration: {selectedPatient.medicalInfo.previousTherapy.duration}</p>
                        )}
                        {selectedPatient.medicalInfo.previousTherapy.reason && (
                          <p className="text-purple-700">Reason: {selectedPatient.medicalInfo.previousTherapy.reason}</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Preferences */}
              {selectedPatient.preferences && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-indigo-600" />
                    Preferences
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-gray-50 p-4 rounded-lg">
                    <div>
                      <p className="text-sm text-gray-500">Preferred Contact</p>
                      <p className="font-medium capitalize">{selectedPatient.preferences.preferredContactMethod || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Session Format</p>
                      <p className="font-medium capitalize">{selectedPatient.preferences.sessionFormat || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Therapist Gender Preference</p>
                      <p className="font-medium capitalize">{selectedPatient.preferences.therapistGenderPreference || 'N/A'}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Guardian Information */}
              {selectedPatient.guardianInfo?.isMinor && selectedPatient.guardianInfo.guardians?.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <UserCheck className="w-5 h-5 text-indigo-600" />
                    Guardian Information (Minor Patient)
                  </h3>
                  {selectedPatient.guardianInfo.guardians.map((guardian, index) => (
                    <div key={index} className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg mb-3">
                      <p className="font-medium text-lg">
                        {guardian.firstName} {guardian.lastName}
                        {guardian.isPrimary && <span className="ml-2 text-xs bg-yellow-200 text-yellow-800 px-2 py-1 rounded">Primary</span>}
                      </p>
                      <p className="text-sm text-gray-600">Relationship: {guardian.relationship}</p>
                      {guardian.phoneNumber && <p className="text-sm text-gray-600">Phone: {guardian.phoneNumber}</p>}
                      {guardian.email && <p className="text-sm text-gray-600">Email: {guardian.email}</p>}
                    </div>
                  ))}
                </div>
              )}

              {/* Registration Date */}
              <div className="pt-4 border-t">
                <p className="text-sm text-gray-500">
                  Patient registered on: <span className="font-medium text-gray-700">{formatDate(selectedPatient.createdAt)}</span>
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
