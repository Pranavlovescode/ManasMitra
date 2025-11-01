"use client";

import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import JournalModule from "@/components/patient_v0/journal-module";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function JournalPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (isLoaded && !user) {
      router.push('/sign-in');
    }
  }, [isLoaded, user, router]);

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-600 mx-auto mb-6"></div>
          <p className="text-gray-600 text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <Card className="mb-8">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl font-bold text-gray-900">
                🌟 Your Journal
              </CardTitle>
              <p className="text-gray-600 mt-2">
                Reflect on your thoughts and emotions with AI-powered insights
              </p>
            </CardHeader>
          </Card>

          {/* Journal Module */}
          <JournalModule userId={user.id} />

          {/* Info Card */}
          <Card className="mt-8">
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div className="p-4">
                  <div className="text-2xl mb-2">🧠</div>
                  <h3 className="font-semibold">AI Analysis</h3>
                  <p className="text-sm text-gray-600">
                    Get insights into your emotional patterns and thought processes
                  </p>
                </div>
                <div className="p-4">
                  <div className="text-2xl mb-2">💡</div>
                  <h3 className="font-semibold">Therapeutic Insights</h3>
                  <p className="text-sm text-gray-600">
                    Receive personalized recommendations and coping strategies
                  </p>
                </div>
                <div className="p-4">
                  <div className="text-2xl mb-2">📈</div>
                  <h3 className="font-semibold">Progress Tracking</h3>
                  <p className="text-sm text-gray-600">
                    Monitor your mental health journey over time
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}