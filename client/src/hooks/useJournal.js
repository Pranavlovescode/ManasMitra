import { useState, useEffect, useCallback } from 'react';

export const useJournal = (userId) => {
  const [journals, setJournals] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [message, setMessage] = useState("");

  const fetchJournals = useCallback(async () => {
    try {
      const res = await fetch(`/api/journal?userId=${userId}`);
      if (res.ok) {
        const data = await res.json();
        setJournals(data);
      }
    } catch (error) {
      console.error("Failed to fetch journals:", error);
    }
  }, [userId]);

  useEffect(() => {
    if (userId) {
      fetchJournals();
    }
  }, [userId, fetchJournals]);

  const createJournal = async (journalData) => {
    setIsLoading(true);
    setMessage("");

    try {
      const res = await fetch("/api/journal", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(journalData),
      });

      if (res.ok) {
        const savedJournal = await res.json();
        setMessage(
          savedJournal.analysis 
            ? "Journal entry saved with AI analysis!" 
            : "Journal entry saved! (Analysis unavailable)"
        );
        await fetchJournals(); // Refresh the list
        return { success: true, journal: savedJournal };
      } else {
        const errorData = await res.json();
        setMessage(errorData.error || "Failed to save entry");
        return { success: false, error: errorData.error };
      }
    } catch (error) {
      console.error("Journal submission error:", error);
      setMessage("An error occurred while saving");
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeJournal = async (journalId) => {
    setIsAnalyzing(true);
    setMessage("");

    try {
      const res = await fetch("/api/journal/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ journalId }),
      });

      if (res.ok) {
        setMessage("Analysis completed!");
        await fetchJournals(); // Refresh to show updated analysis
        return { success: true };
      } else {
        const errorData = await res.json();
        setMessage(errorData.error || "Analysis failed");
        return { success: false, error: errorData.error };
      }
    } catch (error) {
      console.error("Analysis error:", error);
      setMessage("Analysis service unavailable");
      return { success: false, error: error.message };
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearMessage = () => setMessage("");

  return {
    journals,
    isLoading,
    isAnalyzing,
    message,
    createJournal,
    analyzeJournal,
    fetchJournals,
    clearMessage,
  };
};