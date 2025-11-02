"use client";

import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useUserProfile } from "@/hooks/useUserProfile.js";
import ProfileCompletion from "@/app/testing/ProfileCompletion.jsx";

export default function SignUpBasedRedirect() {
  const { user, isLoaded } = useUser();
  const { dbUser, loading, isProfileComplete } = useUserProfile();
  const router = useRouter();
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    const handleRedirect = async () => {
      if (!isLoaded || loading || !user || isRedirecting) return;

      setIsRedirecting(true);

      try {
        // First, try to get user role from Clerk metadata
        let userRole = user?.unsafeMetadata?.role || user?.publicMetadata?.role;

        // If no role in Clerk metadata, check our database
        if (!userRole && dbUser) {
          userRole = dbUser.role;
        }

        // If still no role, try fetching from API
        if (!userRole) {
          try {
            const response = await fetch("/api/users", {
              method: "GET",
              headers: {
                "Content-Type": "application/json",
              },
            });

            if (response.ok) {
              const userData = await response.json();
              userRole = userData?.role;
            }
          } catch (error) {
            console.error("Error fetching user data:", error);
          }
        }

        // If user exists but no role is set, redirect to role selection or profile completion
        if (!userRole) {
          console.log(
            "No role found, redirecting to profile completion or role selection"
          );
          router.push("/sign-up"); // You might want to create a role selection page
          return;
        }

        // Redirect based on role
        switch (userRole) {
          case "therapist":
            router.push("/therapist/onboarding");
            break;
          case "admin":
            router.push("/admin/dashboard");
            break;
          case "patient":
            router.push("/patient-details");
            break;
          default:
            router.push("/patient-details");
            break;            
        }
      } catch (error) {
        console.error("Error during role-based redirect:", error);
        // Default redirect to patient dashboard if there's an error
        router.push("/patient-details");
      }
    };

    handleRedirect();
  }, [user, isLoaded, dbUser, loading, router, isRedirecting]);

  // Show profile completion if user exists but profile is not complete
  if (dbUser && !isProfileComplete) {
    return <ProfileCompletion />;
  }

  // Show loading state while redirecting
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Redirecting to profile completion....</p>
      </div>
    </div>
  );
}
