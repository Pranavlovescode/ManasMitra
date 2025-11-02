"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import PatientList from "@/components/therapist_v0/patient-list";
import PatientDetails from "@/components/therapist_v0/patient-details";
import AnalyticsDashboard from "@/components/therapist_v0/analytics-dashboard";
import AlertsPanel from "@/components/therapist_v0/alerts-panel";
import SessionManagement from "@/components/therapist_v0/session-management";
import { SignOutButton, useUser } from "@clerk/nextjs";
import { useUserProfile } from "@/hooks/useUserProfile";

export default function TherapistDashboard() {
  const router = useRouter();
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [activeTab, setActiveTab] = useState("patients");
  const [isLoading, setIsLoading] = useState(false);
  const user = useUser();
  const { dbUser, loading: profileLoading, isProfileComplete } = useUserProfile();

  // Check if therapist profile is complete
  useEffect(() => {
    if (user.isLoaded && !profileLoading) {
      const userRole = user.user?.publicMetadata?.role || user.user?.unsafeMetadata?.role;
      if (userRole === 'therapist' && !isProfileComplete) {
        router.push('/therapist/onboarding');
      }
    }
  }, [user.isLoaded, profileLoading, isProfileComplete, router, user.user]);
  // console.log(user)
  // useEffect(() => {
  //   const checkAuth = async () => {
  //     const token = localStorage.getItem("token");
  //     if (!token) {
  //       router.push("/auth/login?role=therapist");
  //       return;
  //     }

  //     try {
  //       const res = await fetch("/api/auth/me", {
  //         headers: { Authorization: `Bearer ${token}` },
  //       });

  //       if (!res.ok) {
  //         localStorage.removeItem("token");
  //         router.push("/auth/login?role=therapist");
  //         return;
  //       }

  //       const data = await res.json();
  //       if (data.role !== "therapist") {
  //         router.push("/");
  //         return;
  //       }

  //       setUser(data);
  //     } catch (error) {
  //       router.push("/auth/login?role=therapist");
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

  const handleSelectPatient = (patientId) => {
    setSelectedPatientId(patientId);
    setActiveTab("details"); // Automatically switch to Details tab
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
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">
                  Welcome back,{" "}
                  <span className="text-indigo-600">Dr. {user.user?.firstName}</span>{" "}
                  👨‍⚕️
                </h1>
                <p className="text-gray-600 mt-1">
                  Your professional therapy practice dashboard
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Button
                variant="outline"
                onClick={() => router.push("/therapist/manage-patients")}
                className="bg-indigo-600 text-white hover:bg-indigo-700 border-indigo-600 hover:border-indigo-700 shadow-sm"
              >
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                  />
                </svg>
                Manage Patients
              </Button>
              <SignOutButton>
                <Button
                  variant="outline"
                  onClick={handleLogout}
                  className="bg-white/80 hover:bg-white border-gray-200 hover:border-gray-300 shadow-sm"
                >
                  <svg
                    className="w-4 h-4 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                    />
                  </svg>
                  Logout
                </Button>
              </SignOutButton>
            </div>
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
                  <p className="text-sm text-gray-600 mb-1">Active Patients</p>
                  <p className="text-2xl font-bold text-gray-800">24</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-green-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                    />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Today's Sessions</p>
                  <p className="text-2xl font-bold text-gray-800">6</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-blue-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Priority Alerts</p>
                  <p className="text-2xl font-bold text-gray-800">3</p>
                </div>
                <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-orange-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                    />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">This Week</p>
                  <p className="text-2xl font-bold text-gray-800">18 hours</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-purple-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Card className="bg-white/80 backdrop-blur-sm border-white/20 shadow-xl">
          <CardContent className="p-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <div className="flex justify-center">
                <TabsList className="grid grid-cols-5 bg-gray-100/80 p-1 rounded-xl shadow-inner">
                  <TabsTrigger
                    value="patients"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">👥</span>
                      <span className="hidden sm:inline">Patients</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger
                    value="details"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">👤</span>
                      <span className="hidden sm:inline">Details</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger
                    value="analytics"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">📊</span>
                      <span className="hidden sm:inline">Analytics</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger
                    value="alerts"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">🚨</span>
                      <span className="hidden sm:inline">Alerts</span>
                    </span>
                  </TabsTrigger>
                  <TabsTrigger
                    value="sessions"
                    className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-lg">🗓️</span>
                      <span className="hidden sm:inline">Sessions</span>
                    </span>
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="patients" className="space-y-4 mt-8">
                <PatientList
                  therapistId={user.id}
                  onSelectPatient={handleSelectPatient}
                />
              </TabsContent>

              <TabsContent value="details" className="space-y-4 mt-8">
                {selectedPatientId ? (
                  <PatientDetails
                    patientId={selectedPatientId}
                    therapistId={user.id}
                  />
                ) : (
                  <Card className="bg-gray-50/50">
                    <CardContent className="p-12 text-center">
                      <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg
                          className="w-8 h-8 text-gray-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                          />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-700 mb-2">
                        No Patient Selected
                      </h3>
                      <p className="text-gray-500">
                        Please select a patient from the Patients tab to view
                        their details
                      </p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="analytics" className="space-y-4 mt-8">
                {selectedPatientId ? (
                  <AnalyticsDashboard patientId={selectedPatientId} />
                ) : (
                  <Card className="bg-gray-50/50">
                    <CardContent className="p-12 text-center">
                      <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg
                          className="w-8 h-8 text-gray-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                          />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-700 mb-2">
                        No Patient Selected
                      </h3>
                      <p className="text-gray-500">
                        Please select a patient from the Patients tab to view
                        their analytics
                      </p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="alerts" className="space-y-4 mt-8">
                <AlertsPanel therapistId={user.id} />
              </TabsContent>

              <TabsContent value="sessions" className="space-y-4 mt-8">
                <SessionManagement therapistId={user.id} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
