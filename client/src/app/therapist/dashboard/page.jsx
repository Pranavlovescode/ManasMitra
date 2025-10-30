"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import PatientList from "@/components/therapist/patient-list";
import PatientDetails from "@/components/therapist/patient-details";
import AnalyticsDashboard from "@/components/therapist/analytics-dashboard";
import AlertsPanel from "@/components/therapist/alerts-panel";
import SessionManagement from "@/components/therapist/session-management";

export default function TherapistDashboard() {
  const router = useRouter();
  const [user, setUser] = useState(null); // Type <any> removed
  const [selectedPatientId, setSelectedPatientId] = useState(null); // Type <string | null> removed
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        router.push("/auth/login?role=therapist");
        return;
      }

      try {
        const res = await fetch("/api/auth/me", {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!res.ok) {
          localStorage.removeItem("token");
          router.push("/auth/login?role=therapist");
          return;
        }

        const data = await res.json();
        if (data.role !== "therapist") {
          router.push("/");
          return;
        }

        setUser(data);
      } catch (error) {
        router.push("/auth/login?role=therapist");
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/");
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Dr. {user.lastName}</h1>
            <p className="text-muted-foreground">Therapist Dashboard</p>
          </div>
          <Button variant="outline" onClick={handleLogout}>
            Logout
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="patients" className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="patients">Patients</TabsTrigger>
            <TabsTrigger value="details">Patient Details</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
            <TabsTrigger value="sessions">Sessions</TabsTrigger>
          </TabsList>

          <TabsContent value="patients" className="space-y-4">
            <PatientList
              therapistId={user.id}
              onSelectPatient={setSelectedPatientId}
            />
          </TabsContent>

          <TabsContent value="details" className="space-y-4">
            {selectedPatientId ? (
              <PatientDetails
                patientId={selectedPatientId}
                therapistId={user.id}
              />
            ) : (
              <div className="p-8 text-center text-muted-foreground">
                Select a patient to view details
              </div>
            )}
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            {selectedPatientId ? (
              <AnalyticsDashboard patientId={selectedPatientId} />
            ) : (
              <div className="p-8 text-center text-muted-foreground">
                Select a patient to view analytics
              </div>
            )}
          </TabsContent>

          <TabsContent value="alerts" className="space-y-4">
            <AlertsPanel therapistId={user.id} />
          </TabsContent>

          <TabsContent value="sessions" className="space-y-4">
            <SessionManagement therapistId={user.id} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
