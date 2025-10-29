'use client';

import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';

export function useUserProfile() {
  const { user, isLoaded } = useUser();
  const [dbUser, setDbUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!isLoaded || !user) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch('/api/users', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const userData = await response.json();
          setDbUser(userData);
        } else if (response.status === 404) {
          // User not found in database yet
          setDbUser(null);
        } else {
          throw new Error('Failed to fetch user profile');
        }
      } catch (err) {
        console.error('Error fetching user profile:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [user, isLoaded]);

  // Check if profile is complete based on role
  const isProfileComplete = dbUser ? checkProfileCompleteness(dbUser) : false;

  return {
    dbUser,
    loading,
    error,
    isProfileComplete,
    refetch: () => {
      setLoading(true);
      fetchUserProfile();
    }
  };
}

function checkProfileCompleteness(user) {
  if (!user) return false;

  // Common required fields
  const commonFields = ['firstName', 'lastName', 'email', 'role'];
  const hasCommonFields = commonFields.every(field => user[field] && user[field].trim() !== '');

  if (!hasCommonFields) return false;

  // Role-specific required fields
  if (user.role === 'patient') {
    const patientFields = ['dateOfBirth', 'emergencyContact'];
    return patientFields.every(field => {
      if (field === 'emergencyContact') {
        return user[field] && user[field].name && user[field].phone;
      }
      return user[field] && user[field].toString().trim() !== '';
    });
  } else if (user.role === 'therapist') {
    const therapistFields = ['licenseNumber', 'specializations'];
    return therapistFields.every(field => {
      if (field === 'specializations') {
        return user[field] && Array.isArray(user[field]) && user[field].length > 0;
      }
      return user[field] && user[field].toString().trim() !== '';
    });
  }

  return false;
}