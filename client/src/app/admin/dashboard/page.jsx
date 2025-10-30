"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui_1/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import TherapistVerification from "@/components/admin/therapist-verification";
import UserManagement from "@/components/admin/user-management";
import SystemAnalytics from "@/components/admin/system-analytics";
import ComplianceReports from "@/components/admin/compliance-reports";

export default function AdminDashboard() {
  const router = useRouter();
  const [user, setUser] = useState();
  const [isLoading, setIsLoading] = useState(false);

  // useEffect(() => {
  //   const checkAuth = async () => {
  //     const token = localStorage.getItem("token");
  //     if (!token) {
  //       router.push("/auth/login?role=admin");
  //       return;
  //     }

  //     try {
  //       const res = await fetch("/api/auth/me", {
  //         headers: { Authorization: `Bearer ${token}` },
  //       });

  //       if (!res.ok) {
  //         localStorage.removeItem("token");
  //         router.push("/auth/login?role=admin");
  //         return;
  //       }

  //       const data = await res.json();
  //       if (data.role !== "admin") {
  //         router.push("/");
  //         return;
  //       }

  //       setUser(data);
  //     } catch (error) {
  //       router.push("/auth/login?role=admin");
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
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // if (!user) {
  //   return null;
  // }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Admin Dashboard</h1>
            <p className="text-muted-foreground">
              System Management & Oversight
            </p>
          </div>
          <Button variant="outline" onClick={handleLogout}>
            Logout
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="verification" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="verification">Verification</TabsTrigger>
            <TabsTrigger value="users">Users</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="compliance">Compliance</TabsTrigger>
          </TabsList>

          <TabsContent value="verification" className="space-y-4">
            <TherapistVerification />
          </TabsContent>

          <TabsContent value="users" className="space-y-4">
            <UserManagement />
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            <SystemAnalytics />
          </TabsContent>

          <TabsContent value="compliance" className="space-y-4">
            <ComplianceReports />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
