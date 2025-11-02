"use client";

import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";

export function useUserProfile() {
  const { user, isLoaded } = useUser();
  const [dbUser, setDbUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchUserProfile = async () => {
    // Wait for Clerk to fully load
    if (!isLoaded) {
      console.log('🔄 [useUserProfile] Waiting for Clerk to load...');
      return;
    }

    // If no user is signed in, stop loading
    if (!user) {
      console.log('ℹ️ [useUserProfile] No user signed in');
      setLoading(false);
      setDbUser(null);
      setError(null);
      return;
    }

    // Check if user has a valid ID (indicates session is established)
    if (!user.id) {
      console.log('⚠️ [useUserProfile] User object exists but no ID found, waiting...');
      return;
    }

    console.log('🔍 [useUserProfile] Fetching profile for user:', user.id);

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/users', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));

        if (response.status === 401) {
          console.warn('User not authenticated. Please sign in.');
          setError('Please sign in to continue');
          setDbUser(null);
          return;
        }

        console.error('Profile fetch error:', response.status, errorData);
        throw new Error(errorData.error || 'Failed to fetch user profile');
      }

      const userData = await response.json();
      console.log('✅ [useUserProfile] Profile fetched successfully');
      setDbUser(userData);
      setError(null);
    } catch (err) {
      console.error('❌ [useUserProfile] Error fetching user profile:', err);
      setError(err.message);
      setDbUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Add a small delay to ensure session is fully established
    const timeoutId = setTimeout(fetchUserProfile, 100);
    return () => clearTimeout(timeoutId);
  }, [user, isLoaded]);

  // Check if profile is complete based on role
  const isProfileComplete = dbUser ? checkProfileCompleteness(dbUser) : false;

  return {
    dbUser,
    loading,
    error,
    isProfileComplete,
    refetch: () => {
      console.log('🔄 [useUserProfile] Manual refetch triggered');
      fetchUserProfile();
    },
  };
}

function checkProfileCompleteness(user) {
  if (!user) return false;

  // Common required fields
  const commonFields = ["firstName", "lastName", "email", "role"];
  const hasCommonFields = commonFields.every(
    (field) => user[field] && user[field].trim() !== ""
  );

  if (!hasCommonFields) return false;

  // For both patients and therapists, rely on the profileComplete flag
  // This flag is set to true when:
  // - Patients complete their detailed information via /api/patients
  // - Therapists complete their onboarding process
  return user.profileComplete === true;
}
