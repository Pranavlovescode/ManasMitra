import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { apiClient } from '@/lib/api';

export function useUserProfile() {
  const { user: clerkUser, isLoaded } = useUser();
  const [dbUser, setDbUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchUserProfile() {
      if (!isLoaded || !clerkUser) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await apiClient.getCurrentUser();
        setDbUser(response.user);
        setError(null);
      } catch (err) {
        console.error('Error fetching user profile:', err);
        setError(err.message);
        setDbUser(null);
      } finally {
        setLoading(false);
      }
    }

    fetchUserProfile();
  }, [clerkUser, isLoaded]);

  const updateProfile = async (profileData) => {
    try {
      const response = await apiClient.updateUser(profileData);
      setDbUser(response.user);
      return response.user;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const completeProfile = async (profileData) => {
    try {
      const response = await apiClient.completeProfile(profileData);
      setDbUser(response.user);
      return response.user;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  return {
    clerkUser,
    dbUser,
    loading,
    error,
    updateProfile,
    completeProfile,
    isProfileComplete: dbUser?.profileComplete || false,
    userRole: dbUser?.role || clerkUser?.unsafeMetadata?.role || 'patient'
  };
}

export function useTherapistData() {
  const [patients, setPatients] = useState([]);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getPatients();
      setPatients(response.patients);
      setError(null);
    } catch (err) {
      console.error('Error fetching patients:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchDashboardData = async () => {
    try {
      const response = await apiClient.getTherapistDashboard();
      setDashboardData(response.dashboardData);
      setError(null);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchPatients();
    fetchDashboardData();
  }, []);

  return {
    patients,
    dashboardData,
    loading,
    error,
    refetchPatients: fetchPatients,
    refetchDashboardData: fetchDashboardData
  };
}