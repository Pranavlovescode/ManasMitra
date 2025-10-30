"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function TherapistVerification() {
  const [therapists, setTherapists] = useState([]); // Type <any[]> removed
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchTherapists();
  }, []);

  const fetchTherapists = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/admin/therapists", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setTherapists(data);
      }
    } catch (error) {
      console.error("Failed to fetch therapists:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerify = async (therapistId) => {
    // Type :string removed
    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`/api/admin/verify-therapist/${therapistId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        fetchTherapists();
      }
    } catch (error) {
      console.error("Failed to verify therapist:", error);
    }
  };

  const handleReject = async (therapistId) => {
    // Type :string removed
    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`/api/admin/reject-therapist/${therapistId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        fetchTherapists();
      }
    } catch (error) {
      console.error("Failed to reject therapist:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading therapists...
      </div>
    );
  }

  const pendingTherapists = therapists.filter((t) => !t.verified);
  const verifiedTherapists = therapists.filter((t) => t.verified);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Pending Verification</CardTitle>
          <CardDescription>
            Review and verify therapist credentials
          </CardDescription>
        </CardHeader>
        <CardContent>
          {pendingTherapists.length > 0 ? (
            <div className="space-y-3">
              {pendingTherapists.map((therapist) => (
                <div
                  key={therapist._id}
                  className="flex items-center justify-between p-4 bg-muted rounded-lg"
                >
                  <div>
                    <p className="font-semibold">
                      Dr. {therapist.firstName} {therapist.lastName}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {therapist.email}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      License: {therapist.licenseNumber}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleVerify(therapist._id)}
                    >
                      Verify
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleReject(therapist._id)}
                    >
                      Reject
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No pending verifications
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Verified Therapists</CardTitle>
          <CardDescription>
            {verifiedTherapists.length} therapists verified
          </CardDescription>
        </CardHeader>
        <CardContent>
          {verifiedTherapists.length > 0 ? (
            <div className="space-y-3">
              {verifiedTherapists.map((therapist) => (
                <div
                  key={therapist._id}
                  className="flex items-center justify-between p-4 bg-muted rounded-lg"
                >
                  <div>
                    <p className="font-semibold">
                      Dr. {therapist.firstName} {therapist.lastName}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {therapist.email}
                    </p>
                  </div>
                  <Badge>Verified</Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No verified therapists yet
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
