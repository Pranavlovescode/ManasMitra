"use client";

import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import QuestionnaireForm from "@/components/QuestionnaireForm";
import { Card } from "../../components/ui_1/card";
import { ChevronLeft } from "lucide-react";

export default function AssessmentPage() {
  const { userId } = useAuth();
  const router = useRouter();

  const handleSubmit = async (assessment) => {
    try {
      const response = await fetch("/api/assessments", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(assessment),
      });

      if (!response.ok) {
        throw new Error("Failed to submit assessment");
      }

      // Show success message or redirect
      router.push("/dashboard");
    } catch (error) {
      console.error("Error submitting assessment:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <button
                onClick={() => router.push("/dashboard")}
                className="mr-4 p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100"
              >
                <ChevronLeft className="h-5 w-5" />
              </button>
              <h1 className="text-2xl font-bold text-gray-900">
                Mental Health Assessment
              </h1>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <Card className="bg-white p-6 mb-6 shadow-sm">
            <p className="text-gray-700 mb-4 text-lg">
              Over the last 2 weeks, how often have you been bothered by any of
              the following problems?
            </p>
            <p className="text-sm text-gray-500">
              Please answer all questions honestly. Your responses will help us
              better understand your current state of mental health.
            </p>
          </Card>
          <QuestionnaireForm onSubmit={handleSubmit} />
        </div>
      </main>
    </div>
  );
}
