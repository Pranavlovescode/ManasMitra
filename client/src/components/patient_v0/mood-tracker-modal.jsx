"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../ui/dialog";
import { Textarea } from "../ui/textarea";

const MOODS = [
  { emoji: "😢", label: "Sad", value: "sad", color: "bg-blue-100 hover:bg-blue-200 border-blue-300" },
  { emoji: "😐", label: "Neutral", value: "neutral", color: "bg-gray-100 hover:bg-gray-200 border-gray-300" },
  { emoji: "🙂", label: "Happy", value: "happy", color: "bg-yellow-100 hover:bg-yellow-200 border-yellow-300" },
  { emoji: "😄", label: "Excited", value: "excited", color: "bg-orange-100 hover:bg-orange-200 border-orange-300" },
  { emoji: "😍", label: "Loved", value: "loved", color: "bg-pink-100 hover:bg-pink-200 border-pink-300" },
];

export default function MoodTrackerModal({ isOpen, onClose, userId, onMoodSubmitted }) {
  const [selectedMood, setSelectedMood] = useState(null);
  const [intensity, setIntensity] = useState(5);
  const [notes, setNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleSubmit = async () => {
    if (!selectedMood) {
      setMessage("Please select a mood");
      return;
    }

    setIsLoading(true);
    setMessage("");

    try {
      // Since we're using Clerk, we'll need to make the API call to your mood endpoint
      // For now, I'll use a placeholder API call structure
      const response = await fetch("/api/mood", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId,
          mood: selectedMood,
          intensity,
          notes,
          timestamp: new Date().toISOString(),
        }),
      });

      if (response.ok) {
        setMessage("Mood logged successfully!");
        setTimeout(() => {
          // Reset form
          setSelectedMood(null);
          setIntensity(5);
          setNotes("");
          setMessage("");
          // Close modal and notify parent
          onMoodSubmitted?.(selectedMood, intensity);
          onClose();
        }, 1500);
      } else {
        setMessage("Failed to log mood. Please try again.");
      }
    } catch (error) {
      console.error("Error logging mood:", error);
      setMessage("An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSkip = () => {
    // Reset form and close
    setSelectedMood(null);
    setIntensity(5);
    setNotes("");
    setMessage("");
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-center text-gray-800">
            How are you feeling today? 🌟
          </DialogTitle>
          <DialogDescription className="text-center text-gray-600">
            Let's track your mood to help us understand your mental health journey better
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 py-4">
          {/* Mood Selection */}
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-700">Select your current mood:</label>
            <div className="grid grid-cols-5 gap-3">
              {MOODS.map((mood) => (
                <button
                  key={mood.value}
                  onClick={() => setSelectedMood(mood.value)}
                  className={`flex flex-col items-center gap-2 p-3 rounded-xl border-2 transition-all duration-200 ${
                    selectedMood === mood.value
                      ? "border-indigo-400 bg-indigo-50 scale-105 shadow-md"
                      : `${mood.color} border-transparent hover:scale-105`
                  }`}
                >
                  <span className="text-3xl">{mood.emoji}</span>
                  <span className="text-xs font-medium text-gray-700">{mood.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Intensity Slider */}
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-700">
              How intense is this feeling? ({intensity}/10)
            </label>
            <div className="space-y-2">
              <input
                type="range"
                min="1"
                max="10"
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
                className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #6366f1 0%, #6366f1 ${intensity * 10}%, #e5e7eb ${intensity * 10}%, #e5e7eb 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Low</span>
                <span className="font-medium text-indigo-600">{intensity}/10</span>
                <span>High</span>
              </div>
            </div>
          </div>

          {/* Notes */}
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-700">
              Any additional thoughts? (Optional)
            </label>
            <Textarea
              placeholder="What's on your mind? Any specific thoughts or events that influenced your mood today?"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="min-h-20 resize-none border-2 border-gray-200 focus:border-indigo-400 rounded-lg"
            />
          </div>

          {/* Message */}
          {message && (
            <div
              className={`p-3 rounded-lg text-sm text-center font-medium ${
                message.includes("successfully")
                  ? "bg-green-100 text-green-800 border border-green-200"
                  : "bg-red-100 text-red-800 border border-red-200"
              }`}
            >
              {message}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-2">
            <Button
              variant="outline"
              onClick={handleSkip}
              className="flex-1 border-2 border-gray-200 hover:border-gray-300"
              disabled={isLoading}
            >
              Skip for now
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={isLoading || !selectedMood}
              className="flex-1 bg-linear-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold shadow-lg disabled:opacity-50"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Logging...
                </div>
              ) : (
                "Log My Mood"
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
