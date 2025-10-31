"use client";

import { useState, useEffect } from 'react';

export function useDailyMoodTracker(userId) {
  const [showMoodModal, setShowMoodModal] = useState(false);
  const [hasMoodToday, setHasMoodToday] = useState(false);
  const [isCheckingMood, setIsCheckingMood] = useState(true);

  useEffect(() => {
    if (!userId) return;

    checkTodayMood();
  }, [userId]);

  const checkTodayMood = async () => {
    try {
      setIsCheckingMood(true);
      
      // Check if user has already logged mood today
      const today = new Date().toISOString().split('T')[0]; // Get YYYY-MM-DD format
      
      // Check localStorage first for quick response
      const lastMoodDate = localStorage.getItem(`lastMoodDate_${userId}`);
      const todayMoodLogged = localStorage.getItem(`todayMoodLogged_${userId}`);
      
      if (lastMoodDate === today && todayMoodLogged === 'true') {
        setHasMoodToday(true);
        setShowMoodModal(false);
        setIsCheckingMood(false);
        return;
      }

      // If not in localStorage, check with API
      const response = await fetch(`/api/mood/today?userId=${userId}&date=${today}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        const hasLoggedToday = data.hasMoodToday || false;
        
        setHasMoodToday(hasLoggedToday);
        
        // If user hasn't logged mood today, show modal after a short delay
        if (!hasLoggedToday) {
          setTimeout(() => {
            setShowMoodModal(true);
          }, 2000); // Show modal after 2 seconds
        }
        
        // Update localStorage
        if (hasLoggedToday) {
          localStorage.setItem(`lastMoodDate_${userId}`, today);
          localStorage.setItem(`todayMoodLogged_${userId}`, 'true');
        }
      } else {
        // If API call fails, show modal anyway (better to ask than miss)
        console.warn('Failed to check today mood status, showing modal as fallback');
        setTimeout(() => {
          setShowMoodModal(true);
        }, 2000);
      }
    } catch (error) {
      console.error('Error checking today mood:', error);
      // On error, show modal as fallback
      setTimeout(() => {
        setShowMoodModal(true);
      }, 2000);
    } finally {
      setIsCheckingMood(false);
    }
  };

  const handleMoodSubmitted = (mood, intensity) => {
    const today = new Date().toISOString().split('T')[0];
    
    // Update local state
    setHasMoodToday(true);
    setShowMoodModal(false);
    
    // Update localStorage
    localStorage.setItem(`lastMoodDate_${userId}`, today);
    localStorage.setItem(`todayMoodLogged_${userId}`, 'true');
    
    // Also store the actual mood data for quick reference
    localStorage.setItem(`todayMood_${userId}`, JSON.stringify({
      mood,
      intensity,
      date: today,
      timestamp: new Date().toISOString()
    }));
  };

  const handleModalClose = () => {
    setShowMoodModal(false);
  };

  const getTodayMood = () => {
    try {
      const todayMoodData = localStorage.getItem(`todayMood_${userId}`);
      return todayMoodData ? JSON.parse(todayMoodData) : null;
    } catch {
      return null;
    }
  };

  const showMoodModalManually = () => {
    setShowMoodModal(true);
  };

  return {
    showMoodModal,
    hasMoodToday,
    isCheckingMood,
    handleMoodSubmitted,
    handleModalClose,
    getTodayMood,
    showMoodModalManually,
    checkTodayMood
  };
}