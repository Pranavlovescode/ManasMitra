"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui_1/card";
import { Button } from "../ui_1/button";

export default function AlertsPanel({ therapistId }) {
  // Type annotation removed
  const [alerts, setAlerts] = useState([]); // Type <any[]> removed
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [therapistId]);

  const fetchAlerts = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`/api/alerts?therapistId=${therapistId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setAlerts(data);
      }
    } catch (error) {
      console.error("Failed to fetch alerts:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDismissAlert = async (alertId) => {
    // Type :string removed
    try {
      const token = localStorage.getItem("token");
      await fetch(`/api/alerts/${alertId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      setAlerts(alerts.filter((a) => a._id !== alertId));
    } catch (error) {
      console.error("Failed to dismiss alert:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">Loading alerts...</div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alerts & Notifications</CardTitle>
        <CardDescription>
          High-risk patients and important updates
        </CardDescription>
      </CardHeader>
      <CardContent>
        {alerts.length > 0 ? (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert._id}
                className={`p-4 rounded-lg border-l-4 ${
                  alert.severity === "high"
                    ? "bg-red-50 border-red-500"
                    : alert.severity === "medium"
                    ? "bg-yellow-50 border-yellow-500"
                    : "bg-blue-50 border-blue-500"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-semibold">{alert.title}</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {alert.message}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Patient: {alert.patientName} •{" "}
                      {new Date(alert.createdAt).toLocaleString()}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleDismissAlert(alert._id)}
                  >
                    Dismiss
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-center text-muted-foreground py-8">
            No active alerts
          </p>
        )}
      </CardContent>
    </Card>
  );
}
