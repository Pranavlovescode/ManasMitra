"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import MoodTracker from "@/components/patient_v0/mood-tracker";
import JournalModule from "@/components/patient_v0/journal-module";
import ChatbotAssistant from "@/components/patient_v0/chatbot-assistant";
import AssessmentModule from "@/components/patient_v0/assessment-module";
import AppointmentBooking from "@/components/patient_v0/appointment-booking";
import { useUser } from "@clerk/nextjs";

export default function PatientDashboard() {
  const router = useRouter();
  // const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const user = useUser();

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

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-600 mx-auto mb-6"></div>
          <p className="text-gray-600 text-lg">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  // if (!user) {
  //   return null;
  // }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="backdrop-blur-md bg-white/80 border-b border-white/20 shadow-sm">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-linear-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">
                  Welcome back, <span className="text-indigo-600">{user.firstName}</span> 👋
                </h1>
                <p className="text-gray-600 mt-1">Continue your mental health journey with us</p>
              </div>
            </div>
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
      </header>

      {/* Quick Stats */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Today's Mood</p>
                  <p className="text-2xl font-bold text-gray-800">😊 Happy</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
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

        {/* Main Content */}
        <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-xl">
          <CardContent className="p-8">
            <Tabs defaultValue="mood" className="space-y-6">
              <div className="flex justify-center">
                <TabsList className="grid grid-cols-5 bg-gray-100/80 p-1 rounded-xl shadow-inner">
                  <TabsTrigger 
                    value="mood" 
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">😊</span>
                      <span className="hidden sm:inline">Mood</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="journal"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"  
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">📖</span>
                      <span className="hidden sm:inline">Journal</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="chatbot"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">🤖</span>
                      <span className="hidden sm:inline">Chatbot</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="assessment"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">📋</span>
                      <span className="hidden sm:inline">Assessment</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="appointments"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">📅</span>
                      <span className="hidden sm:inline">Appointments</span>
                    </span>
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="mood" className="space-y-4 mt-8">
                <MoodTracker userId={user.id} />
              </TabsContent>

              <TabsContent value="journal" className="space-y-4 mt-8">
                <JournalModule userId={user.id} />
              </TabsContent>

              <TabsContent value="chatbot" className="space-y-4 mt-8">
                <ChatbotAssistant userId={user.id} />
              </TabsContent>

              <TabsContent value="assessment" className="space-y-4 mt-8">
                <AssessmentModule userId={user.id} />
              </TabsContent>

              <TabsContent value="appointments" className="space-y-4 mt-8">
                <AppointmentBooking userId={user.id} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
