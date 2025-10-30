import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';

export function usePatientData() {
  const { user, isLoaded } = useUser();
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasDetails, setHasDetails] = useState(false);

  const fetchPatientData = async () => {
    if (!user) return;
    
    try {
      setLoading(true);
      const response = await fetch('/api/patients/profile');
      
      if (response.ok) {
        const data = await response.json();
        setPatientData(data.patient);
        setHasDetails(data.hasDetails);
        setError(null);
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to fetch patient data');
      }
    } catch (err) {
      setError('An unexpected error occurred');
      console.error('Error fetching patient data:', err);
    } finally {
      setLoading(false);
    }
  };

  const updatePatientData = async (data) => {
    try {
      setLoading(true);
      const response = await fetch('/api/patients', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setPatientData(result.patient);
        setHasDetails(true);
        setError(null);
        return { success: true, data: result.patient };
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to update patient data');
        return { success: false, error: errorData.error };
      }
    } catch (err) {
      const errorMessage = 'An unexpected error occurred';
      setError(errorMessage);
      console.error('Error updating patient data:', err);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  };

  const updateExistingPatientData = async (data) => {
    try {
      setLoading(true);
      const response = await fetch('/api/patients', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setPatientData(result.patient);
        setHasDetails(true);
        setError(null);
        return { success: true, data: result.patient };
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to update patient data');
        return { success: false, error: errorData.error };
      }
    } catch (err) {
      const errorMessage = 'An unexpected error occurred';
      setError(errorMessage);
      console.error('Error updating patient data:', err);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isLoaded && user && user.unsafeMetadata?.role === 'patient') {
      fetchPatientData();
    } else if (isLoaded) {
      setLoading(false);
    }
  }, [isLoaded, user]);

  return {
    patientData,
    loading,
    error,
    hasDetails,
    fetchPatientData,
    updatePatientData,
    updateExistingPatientData,
    refetch: fetchPatientData,
  };
}