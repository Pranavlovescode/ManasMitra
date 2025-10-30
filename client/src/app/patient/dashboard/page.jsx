"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import MoodTracker from "@/components/patient/mood-tracker";
import JournalModule from "@/components/patient/journal-module";
import ChatbotAssistant from "@/components/patient/chatbot-assistant";
import AssessmentModule from "@/components/patient/assessment-module";
import AppointmentBooking from "@/components/patient/appointment-booking";

export default function PatientDashboard() {
  const router = useRouter();
  const [user, setUser] = useState < any > null;
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        router.push("/auth/login?role=patient");
        return;
      }

      try {
        const res = await fetch("/api/auth/me", {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!res.ok) {
          localStorage.removeItem("token");
          router.push("/auth/login?role=patient");
          return;
        }

        const data = await res.json();
        if (data.role !== "patient") {
          router.push("/");
          return;
        }

        setUser(data);
      } catch (error) {
        router.push("/auth/login?role=patient");
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
            <h1 className="text-2xl font-bold">Welcome, {user.firstName}</h1>
            <p className="text-muted-foreground">Your mental health journey</p>
          </div>
          <Button variant="outline" onClick={handleLogout}>
            Logout
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="mood" className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="mood">Mood</TabsTrigger>
            <TabsTrigger value="journal">Journal</TabsTrigger>
            <TabsTrigger value="chatbot">Chatbot</TabsTrigger>
            <TabsTrigger value="assessment">Assessment</TabsTrigger>
            <TabsTrigger value="appointments">Appointments</TabsTrigger>
          </TabsList>

          <TabsContent value="mood" className="space-y-4">
            <MoodTracker userId={user.id} />
          </TabsContent>

          <TabsContent value="journal" className="space-y-4">
            <JournalModule userId={user.id} />
          </TabsContent>

          <TabsContent value="chatbot" className="space-y-4">
            <ChatbotAssistant userId={user.id} />
          </TabsContent>

          <TabsContent value="assessment" className="space-y-4">
            <AssessmentModule userId={user.id} />
          </TabsContent>

          <TabsContent value="appointments" className="space-y-4">
            <AppointmentBooking userId={user.id} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
