'use client';

import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';

export default function TherapistAssignmentTest() {
  const { user, isLoaded } = useUser();
  const [availableTherapists, setAvailableTherapists] = useState([]);
  const [selectedTherapist, setSelectedTherapist] = useState('');
  const [currentAssignment, setCurrentAssignment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch available therapists
  useEffect(() => {
    const fetchTherapists = async () => {
      try {
        const response = await fetch('/api/therapists/available');
        if (response.ok) {
          const data = await response.json();
          setAvailableTherapists(data.therapists || []);
        } else {
          console.error('Failed to fetch therapists:', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching therapists:', error);
      }
    };

    if (isLoaded && user) {
      fetchTherapists();
    }
  }, [isLoaded, user]);

  // Fetch current therapist assignment
  useEffect(() => {
    const fetchCurrentAssignment = async () => {
      try {
        const response = await fetch('/api/patients/therapist');
        if (response.ok) {
          const data = await response.json();
          setCurrentAssignment(data);
          if (data.therapist) {
            setSelectedTherapist(data.therapist._id);
          }
        }
      } catch (error) {
        console.error('Error fetching current assignment:', error);
      }
    };

    if (isLoaded && user && user.unsafeMetadata?.role === 'patient') {
      fetchCurrentAssignment();
    }
  }, [isLoaded, user]);

  const handleAssignTherapist = async () => {
    setLoading(true);
    setMessage('');

    try {
      const response = await fetch('/api/patients/therapist', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          therapistId: selectedTherapist || null
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentAssignment(data);
        setMessage('Therapist assignment updated successfully!');
      } else {
        const error = await response.json();
        setMessage(`Error: ${error.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const seedTherapists = async () => {
    setLoading(true);
    setMessage('');

    try {
      const response = await fetch('/api/testing/seed-therapists', {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        setMessage('Test therapists created successfully! Refresh the page to see them.');
      } else {
        const error = await response.json();
        setMessage(`Error: ${error.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!isLoaded) {
    return <div>Loading...</div>;
  }

  if (!user) {
    return <div>Please sign in to test therapist assignment.</div>;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Therapist Assignment Test</h1>
      
      <div className="space-y-6">
        {/* Current User Info */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Current User</h2>
          <p>Name: {user.firstName} {user.lastName}</p>
          <p>Email: {user.emailAddresses?.[0]?.emailAddress}</p>
          <p>Role: {user.unsafeMetadata?.role || 'Not set'}</p>
        </div>

        {/* Current Assignment */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Current Therapist Assignment</h2>
          {currentAssignment ? (
            <div>
              {currentAssignment.hasAssignedTherapist ? (
                <div>
                  <p><strong>Assigned to:</strong> {currentAssignment.therapist.name}</p>
                  <p><strong>Email:</strong> {currentAssignment.therapist.email}</p>
                </div>
              ) : (
                <p>No therapist currently assigned</p>
              )}
            </div>
          ) : (
            <p>Loading current assignment...</p>
          )}
        </div>

        {/* Available Therapists */}
        <div className="bg-white border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">Available Therapists</h2>
          
          {availableTherapists.length === 0 ? (
            <div>
              <p className="text-gray-600 mb-4">No therapists available. You may need to seed some test data.</p>
              <button
                onClick={seedTherapists}
                disabled={loading}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
              >
                {loading ? 'Creating...' : 'Create Test Therapists'}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label htmlFor="therapist-select" className="block text-sm font-medium mb-2">
                  Select Therapist:
                </label>
                <select
                  id="therapist-select"
                  value={selectedTherapist}
                  onChange={(e) => setSelectedTherapist(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">No therapist assigned</option>
                  {availableTherapists.map((therapist) => (
                    <option key={therapist._id} value={therapist._id}>
                      {therapist.name}
                      {therapist.specializations.length > 0 && 
                        ` - ${therapist.specializations.slice(0, 2).join(', ')}`
                      }
                      {therapist.yearsOfExperience > 0 && 
                        ` (${therapist.yearsOfExperience} years exp.)`
                      }
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleAssignTherapist}
                disabled={loading}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
              >
                {loading ? 'Updating...' : 'Update Assignment'}
              </button>
            </div>
          )}
        </div>

        {/* Therapist Details */}
        {availableTherapists.length > 0 && (
          <div className="bg-white border rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-4">Therapist Details</h2>
            <div className="space-y-4">
              {availableTherapists.map((therapist) => (
                <div key={therapist._id} className="border-l-4 border-blue-500 pl-4">
                  <h3 className="font-semibold">{therapist.name}</h3>
                  <p className="text-sm text-gray-600">{therapist.email}</p>
                  {therapist.specializations.length > 0 && (
                    <p className="text-sm"><strong>Specializations:</strong> {therapist.specializations.join(', ')}</p>
                  )}
                  <p className="text-sm"><strong>Experience:</strong> {therapist.yearsOfExperience} years</p>
                  <p className="text-sm"><strong>Accepting New Patients:</strong> {therapist.acceptingNewPatients ? 'Yes' : 'No'}</p>
                  <p className="text-sm"><strong>Verified:</strong> {therapist.verified ? 'Yes' : 'No'}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Message Display */}
        {message && (
          <div className={`p-4 rounded-lg ${message.includes('Error') ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}