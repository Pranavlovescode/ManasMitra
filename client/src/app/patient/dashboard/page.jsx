"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import DashboardSidebar from "@/components/DashboardSidebar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import JournalModule from "@/components/patient_v0/journal-module";
import ChatbotAssistant from "@/components/patient_v0/chatbot-assistant";
import MultiAssessmentModule from "@/components/patient_v0/multi-assessment-module";
import AppointmentBooking from "@/components/patient_v0/appointment-booking";
import GamesModule from "@/components/patient_v0/games-module";
import MoodTrackerModal from "@/components/patient_v0/mood-tracker-modal";
import { useClerk, useUser } from "@clerk/nextjs";
import { useDailyMoodTracker } from "@/hooks/useDailyMoodTracker";

export default function PatientDashboard() {
  const router = useRouter();
  const { signOut } = useClerk();
  const [isLoading, setIsLoading] = useState(true);
  const { user, isLoaded } = useUser();
  const [activeTab, setActiveTab] = useState("journal");
  const [showGameHistory, setShowGameHistory] = useState(false);
  const [activeGame, setActiveGame] = useState(null);
  const [gameStatus, setGameStatus] = useState("");
  const [savedGameResult, setSavedGameResult] = useState(null);
  const gameIframeRef = useRef(null);

  // Wait for client-side hydration
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Daily mood tracker integration
  const {
    showMoodModal,
    hasMoodToday,
    handleMoodSubmitted,
    handleModalClose,
    getTodayMood,
    showMoodModalManually,
  } = useDailyMoodTracker(user?.id);

  useEffect(() => {
    if (isLoaded) {
      setIsLoading(false);
    }
  }, [isLoaded]);

  // Handle game results from iframe
  useEffect(() => {
    if (!activeGame) {
      setSavedGameResult(null);
      setGameStatus("");
      return;
    }

    // Reset game result when new game starts
    setSavedGameResult(null);
    setGameStatus("");

    function onMessage(ev) {
      const data = ev?.data;
      if (!data || data.type !== "mentalcure:result") return;
      if (data.gameId !== activeGame.id) return;
      
      setGameStatus("Saving result...");
      const payload = data.payload || {};
      const body = {
        gameId: activeGame.id,
        score: Number(payload.score ?? 0),
        metrics: payload,
      };
      
      fetch("/api/games/results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(async (r) => {
          if (r.ok) {
            const j = await r.json();
            setSavedGameResult(j);
            setGameStatus("Saved!");
          } else {
            setGameStatus("Failed to save");
          }
        })
        .catch(() => setGameStatus("Failed to save"))
        .finally(() => {
          setTimeout(() => setGameStatus(""), 2500);
        });
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [activeGame]);

  // useEffect(() => {
  //   const checkAuth = async () => {
  //     const token = localStorage.getItem("token");
  //     if (!token) {
  //       router.push("/auth/login?role=patient");
  //       return;
  //     }

  //     try {
  //       const res = await fetch("/api/auth/me", {
  //         headers: { Authorization: `Bearer ${token}` },
  //       });

  //       if (!res.ok) {
  //         localStorage.removeItem("token");
  //         router.push("/auth/login?role=patient");
  //         return;
  //       }

  //       const data = await res.json();
  //       if (data.role !== "patient") {
  //         router.push("/");
  //         return;
  //       }

  //       setUser(data);
  //     } catch (error) {
  //       router.push("/auth/login?role=patient");
  //     } finally {
  //       setIsLoading(false);
  //     }
  //   };

  //   checkAuth();
  // }, [router]);

  const handleLogout = async () => {
    try {
      // Clear all local storage and session data
      localStorage.clear();
      sessionStorage.clear();
      
      // Sign out from Clerk
      await signOut({ redirectUrl: "/" });
    } catch (error) {
      console.error("Error signing out:", error);
      // Clear storage and redirect even if there's an error
      localStorage.clear();
      sessionStorage.clear();
      router.push("/");
    }
  };

  const sidebarItems = [
    { value: "journal", label: "Journal", icon: "📖" },
    { value: "chatbot", label: "Chatbot", icon: "🤖" },
    { value: "assessment", label: "Assessment", icon: "📋" },
    { value: "games", label: "Games", icon: "🎮" },
    { value: "appointments", label: "Appointments", icon: "📅" },
  ];

  const userInfo = {
    name: user?.firstName || "User",
    role: "Patient",
    initial: user?.firstName?.[0] || "P",
  };

  // Don't render anything until mounted (prevents hydration mismatch)
  if (!mounted || isLoading || !isLoaded) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-600 mx-auto mb-6"></div>
          <p className="text-gray-600 text-lg">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 text-lg">
            User not found. Please sign in again.
          </p>
          <Button onClick={() => router.push("/sign-in")} className="mt-4">
            Sign In
          </Button>
        </div>
      </div>
    );
  }

  const renderContent = () => {
    switch (activeTab) {
      case "journal":
        return <JournalModule userId={user?.id} />;
      case "chatbot":
        return <ChatbotAssistant userId={user?.id} />;
      case "assessment":
        return <MultiAssessmentModule userId={user?.id} />;
      case "games":
        if (activeGame) {
          return (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-800">
                  {activeGame.title}
                </h2>
                <div className="flex items-center gap-3">
                  {gameStatus && (
                    <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-lg">
                      {gameStatus}
                    </span>
                  )}
                  <Button
                    variant="outline"
                    onClick={() => {
                      setActiveGame(null);
                      setSavedGameResult(null);
                      setGameStatus("");
                    }}
                    className="bg-white shadow-sm"
                  >
                    <span className="mr-2">←</span>
                    Back to Games
                  </Button>
                </div>
              </div>

              {savedGameResult && (
                <Card className="bg-green-50 border-green-200">
                  <CardHeader>
                    <CardTitle className="text-green-800">Last Result</CardTitle>
                    <CardDescription>
                      Saved {new Date(savedGameResult.createdAt).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-sm text-gray-700 flex gap-6">
                      <div>
                        Score: <span className="font-semibold">{savedGameResult.score}</span>
                      </div>
                      {typeof savedGameResult.metrics?.accuracy === "number" && (
                        <div>
                          Accuracy:{" "}
                          <span className="font-semibold">
                            {Math.round(savedGameResult.metrics.accuracy * 100)}%
                          </span>
                        </div>
                      )}
                      {typeof savedGameResult.metrics?.avgReactionMs === "number" && (
                        <div>
                          Avg RT:{" "}
                          <span className="font-semibold">
                            {Math.round(savedGameResult.metrics.avgReactionMs)} ms
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="w-full h-[75vh] rounded-xl overflow-hidden border border-gray-200 bg-white shadow-xl">
                <iframe
                  key={activeGame.id}
                  ref={gameIframeRef}
                  src={`/games/mentalcure/${activeGame.id}/index.html?t=${Date.now()}`}
                  className="w-full h-full"
                  title={activeGame.title}
                  allow="fullscreen"
                  sandbox="allow-scripts allow-same-origin allow-forms allow-modals"
                />
              </div>
            </div>
          );
        }
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-800">
                {showGameHistory ? "My Game Scores" : "Available Games"}
              </h2>
              <Button
                variant="outline"
                onClick={() => setShowGameHistory(!showGameHistory)}
                className="bg-white shadow-sm"
              >
                {showGameHistory ? (
                  <>
                    <span className="mr-2">🎮</span>
                    Play Games
                  </>
                ) : (
                  <>
                    <span className="mr-2">📊</span>
                    View Scores
                  </>
                )}
              </Button>
            </div>
            <GamesModule 
              showHistory={showGameHistory} 
              onPlayGame={setActiveGame}
            />
          </div>
        );
      case "appointments":
        return <AppointmentBooking userId={user?.id} />;
      default:
        return <JournalModule userId={user?.id} />;
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex">
      {/* Sidebar */}
      <DashboardSidebar
        items={sidebarItems}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        userInfo={userInfo}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="backdrop-blur-md bg-white/80 border-b border-white/20 shadow-sm sticky top-0 z-30">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-linear-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-800">
                    Welcome back, <span className="text-indigo-600">{user?.firstName || "User"}</span> 👋
                  </h1>
                  <p className="text-gray-600 text-sm mt-1">Continue your mental health journey with us</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {!hasMoodToday && (
                  <Button
                    onClick={showMoodModalManually}
                    className="bg-linear-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white shadow-lg"
                  >
                    <span className="mr-2">😊</span>
                    Log Mood
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={handleLogout}
                  className="bg-white/80 hover:bg-white border-gray-200 hover:border-gray-300 shadow-sm"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  Logout
                </Button>
              </div>
            </div>
          </div>
        </header>

        {/* Quick Stats */}
        <div className="px-6 py-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card
              className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer"
              onClick={showMoodModalManually}
            >
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Today's Mood</p>
                    {(() => {
                      const todayMood = getTodayMood();
                      const moodEmojis = {
                        sad: "😢",
                        neutral: "😐",
                        happy: "🙂",
                        excited: "😄",
                        loved: "😍",
                      };
                      const moodLabels = {
                        sad: "Sad",
                        neutral: "Neutral",
                        happy: "Happy",
                        excited: "Excited",
                        loved: "Loved",
                      };

                      if (todayMood) {
                        return (
                          <p className="text-2xl font-bold text-gray-800">
                            {moodEmojis[todayMood.mood]} {moodLabels[todayMood.mood]}
                          </p>
                        );
                      } else if (hasMoodToday) {
                        return <p className="text-lg font-bold text-gray-800">😊 Logged</p>;
                      } else {
                        return <p className="text-lg font-bold text-gray-500">Not logged yet</p>;
                      }
                    })()}
                  </div>
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                </div>
                {!hasMoodToday && (
                  <p className="text-xs text-gray-500 mt-2">Click to log your mood</p>
                )}
              </CardContent>
            </Card>

            <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Journal Entries</p>
                    <p className="text-2xl font-bold text-gray-800">12</p>
                  </div>
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Sessions</p>
                    <p className="text-2xl font-bold text-gray-800">3</p>
                  </div>
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Streak</p>
                    <p className="text-2xl font-bold text-gray-800">7 days</p>
                  </div>
                  <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                    </svg>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Main Content */}
        <main className="flex-1 px-6 pb-6 overflow-y-auto">
          <div className="max-w-7xl mx-auto">
            {renderContent()}
          </div>
        </main>
      </div>

      {/* Mood Tracker Modal */}
      <MoodTrackerModal
        isOpen={showMoodModal}
        onClose={handleModalClose}
        userId={user?.id}
        onMoodSubmitted={handleMoodSubmitted}
      />
    </div>
  );
}
