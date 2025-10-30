"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { Button } from "@/components/ui_1/button";

export default function RiskAlertBanner() {
  const [alerts, setAlerts] = useState([]); // Type annotation removed
  const [visibleAlerts, setVisibleAlerts] = useState([]); // Type annotation removed

  useEffect(() => {
    fetchRiskAlerts();
    const interval = setInterval(fetchRiskAlerts, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchRiskAlerts = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/alerts/risk", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setAlerts(data);
        setVisibleAlerts(data.slice(0, 3));
      }
    } catch (error) {
      console.error("Failed to fetch risk alerts:", error);
    }
  };

  const handleDismiss = (alertId) => {
    // Type annotation removed
    setVisibleAlerts(visibleAlerts.filter((a) => a.id !== alertId));
  };

  if (visibleAlerts.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2 p-4 bg-background">
      {visibleAlerts.map((alert) => (
        <div
          key={alert.id}
          className={`flex items-start gap-3 p-3 rounded-lg border-l-4 ${
            alert.severity === "high"
              ? "bg-red-50 border-red-500"
              : alert.severity === "medium"
              ? "bg-yellow-50 border-yellow-500"
              : "bg-blue-50 border-blue-500"
          }`}
        >
          <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="font-semibold text-sm">{alert.patientName}</p>
            <p className="text-sm text-muted-foreground">{alert.message}</p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => handleDismiss(alert.id)}
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      ))}
    </div>
  );
}
