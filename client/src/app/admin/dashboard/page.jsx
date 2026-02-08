"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useClerk, useUser } from "@clerk/nextjs";
import { Button } from "@/components/ui_1/button";
import DashboardSidebar from "@/components/DashboardSidebar";
import TherapistVerification from "@/components/admin/therapist-verification";
import UserManagement from "@/components/admin/user-management";
import SystemAnalytics from "@/components/admin/system-analytics";
import ComplianceReports from "@/components/admin/compliance-reports";

export default function AdminDashboard() {
  const router = useRouter();
  const { signOut } = useClerk();
  const { user } = useUser();
  const [activeTab, setActiveTab] = useState("verification");
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
    { value: "verification", label: "Verification", icon: "✓" },
    { value: "users", label: "Users", icon: "👥" },
    { value: "analytics", label: "Analytics", icon: "📊" },
    { value: "compliance", label: "Compliance", icon: "📋" },
  ];

  const userInfo = {
    name: user?.firstName ? `${user.firstName} ${user.lastName || ""}` : "Admin",
    role: "Admin",
    initial: user?.firstName?.[0] || "A",
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

  const renderContent = () => {
    switch (activeTab) {
      case "verification":
        return <TherapistVerification />;
      case "users":
        return <UserManagement />;
      case "analytics":
        return <SystemAnalytics />;
      case "compliance":
        return <ComplianceReports />;
      default:
        return <TherapistVerification />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex">
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
          <div className="px-6 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Admin Dashboard</h1>
              <p className="text-gray-600 text-sm mt-1">
                System Management & Oversight
              </p>
            </div>
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
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-7xl mx-auto">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}
