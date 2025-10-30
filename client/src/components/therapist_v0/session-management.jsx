"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Button } from "../ui/button";

export default function SessionManagement({ therapistId }) {
  // Type annotation removed
  const [sessions, setSessions] = useState([]); // Type <any[]> removed
  const [showCreateSession, setShowCreateSession] = useState(false);
  const [sessionType, setSessionType] = useState(
    // Type union removed
    "individual"
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchSessions();
  }, [therapistId]);

  const fetchSessions = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`/api/webrtc/rooms?therapistId=${therapistId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateSession = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/webrtc/create-room", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          therapistId,
          type: sessionType,
        }),
      });

      if (res.ok) {
        setShowCreateSession(false);
        fetchSessions();
      }
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  const handleJoinSession = (roomId) => {
    // Type :string removed
    window.location.href = `/video-session/${roomId}`;
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading sessions...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {!showCreateSession ? (
        <Card>
          <CardHeader>
            <CardTitle>Create New Session</CardTitle>
            <CardDescription>
              Start a therapy session with your patients
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              onClick={() => setShowCreateSession(true)}
              className="w-full"
            >
              Create Session
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>New Session</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Session Type</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="individual"
                    checked={sessionType === "individual"}
                    onChange={
                      (e) => setSessionType(e.target.value) // 'as' assertion removed
                    }
                  />
                  <span className="text-sm">Individual</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="group"
                    checked={sessionType === "group"}
                    onChange={
                      (e) => setSessionType(e.target.value) // 'as' assertion removed
                    }
                  />
                  <span className="text-sm">Group</span>
                </label>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => setShowCreateSession(false)}
                variant="outline"
                className="flex-1"
              >
                Cancel
              </Button>
              <Button onClick={handleCreateSession} className="flex-1">
                Create
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {sessions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Active Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {sessions.map((session) => (
                <div
                  key={session._id}
                  className="flex items-center justify-between p-4 bg-muted rounded-lg"
                >
                  <div>
                    <p className="font-semibold capitalize">
                      {session.type} Session
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Room ID: {session.roomId} • {session.patientIds.length}{" "}
                      participant(s)
                    </p>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => handleJoinSession(session.roomId)}
                  >
                    {session.active ? "Join" : "View"}
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
