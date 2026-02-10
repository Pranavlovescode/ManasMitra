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
import { Badge } from "../ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "../ui/dialog";
import { Textarea } from "../ui/textarea";
import { AlertCircle, AlertTriangle, Info, X, Eye, CheckCircle2, MessageSquare } from "lucide-react";

export default function AlertsPanel({ therapistId }) {
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState("active"); // "active" or "all"
  const [addressDialog, setAddressDialog] = useState({ open: false, alert: null, notes: "" });

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [therapistId]);

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`/api/alerts`);
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
    try {
      await fetch(`/api/alerts/${alertId}`, {
        method: "DELETE",
      });
      setAlerts(alerts.filter((a) => a._id !== alertId));
    } catch (error) {
      console.error("Failed to dismiss alert:", error);
    }
  };

  const handleAddressAlert = async () => {
    if (!addressDialog.alert) return;

    try {
      const res = await fetch(`/api/alerts/${addressDialog.alert._id}/address`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes: addressDialog.notes }),
      });

      if (res.ok) {
        // Update local state
        setAlerts(alerts.map(a => 
          a._id === addressDialog.alert._id 
            ? { ...a, addressed: true, addressedAt: new Date(), notes: addressDialog.notes }
            : a
        ));
        setAddressDialog({ open: false, alert: null, notes: "" });
      }
    } catch (error) {
      console.error("Failed to address alert:", error);
    }
  };

  const getAlertIcon = (severity) => {
    switch (severity) {
      case "high":
        return <AlertCircle className="w-5 h-5" />;
      case "medium":
        return <AlertTriangle className="w-5 h-5" />;
      default:
        return <Info className="w-5 h-5" />;
    }
  };

  const getAlertColor = (severity) => {
    switch (severity) {
      case "high":
        return "bg-red-50 border-red-500 text-red-900";
      case "medium":
        return "bg-orange-50 border-orange-500 text-orange-900";
      default:
        return "bg-blue-50 border-blue-500 text-blue-900";
    }
  };

  const getSeverityBadge = (severity) => {
    const colors = {
      high: "bg-red-600 text-white",
      medium: "bg-orange-600 text-white",
      low: "bg-blue-600 text-white",
    };
    return colors[severity] || colors.low;
  };

  // Filter alerts based on selected filter
  const filteredAlerts = filter === "active" 
    ? alerts.filter(a => !a.addressed && !a.dismissed)
    : alerts;

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">Loading alerts...</div>
    );
  }

  return (
    <>
      <Card className="border-2">
        <CardHeader className="bg-gradient-to-r from-red-50 to-orange-50">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="w-6 h-6 text-red-600" />
                Alerts & Notifications
              </CardTitle>
              <CardDescription>
                High-risk patients and important updates
              </CardDescription>
            </div>
            {/* Filter Toggle */}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={filter === "active" ? "default" : "outline"}
                onClick={() => setFilter("active")}
              >
                Active ({alerts.filter(a => !a.addressed && !a.dismissed).length})
              </Button>
              <Button
                size="sm"
                variant={filter === "all" ? "default" : "outline"}
                onClick={() => setFilter("all")}
              >
                All ({alerts.length})
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          {filteredAlerts.length > 0 ? (
            <div className="space-y-3">
              {filteredAlerts.map((alert) => (
                <div
                  key={alert._id}
                  className={`p-4 rounded-lg border-l-4 ${getAlertColor(alert.severity)} transition-all hover:shadow-md ${
                    alert.addressed ? 'opacity-75' : ''
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {alert.addressed ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600" />
                      ) : (
                        getAlertIcon(alert.severity)
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="font-semibold">{alert.title}</p>
                          <Badge className={getSeverityBadge(alert.severity)}>
                            {alert.severity?.toUpperCase()}
                          </Badge>
                          {alert.addressed && (
                            <Badge className="bg-green-600 text-white">
                              ✓ ADDRESSED
                            </Badge>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {alert.message}
                      </p>
                      
                      {/* Show notes if addressed */}
                      {alert.addressed && alert.notes && (
                        <div className="mt-3 p-3 bg-white/50 rounded border border-gray-300">
                          <div className="flex items-center gap-2 mb-1">
                            <MessageSquare className="w-4 h-4 text-gray-600" />
                            <span className="text-xs font-semibold text-gray-700">Therapist Notes:</span>
                          </div>
                          <p className="text-sm text-gray-700">{alert.notes}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            Addressed: {new Date(alert.addressedAt).toLocaleString()}
                          </p>
                        </div>
                      )}

                      <div className="flex items-center justify-between mt-3">
                        <p className="text-xs text-muted-foreground">
                          Patient: <span className="font-medium">{alert.patientName}</span> •{" "}
                          {new Date(alert.createdAt).toLocaleString()}
                        </p>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 text-xs"
                            onClick={() => {
                              window.location.href = `/therapist/patients/${alert.patientId}`;
                            }}
                          >
                            <Eye className="w-3 h-3 mr-1" />
                            View
                          </Button>
                          {!alert.addressed && (
                            <Button
                              size="sm"
                              variant="default"
                              className="h-7 text-xs bg-green-600 hover:bg-green-700"
                              onClick={() => setAddressDialog({ open: true, alert, notes: "" })}
                            >
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Address
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs"
                            onClick={() => handleDismissAlert(alert._id)}
                          >
                            <X className="w-3 h-3 mr-1" />
                            Dismiss
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-100 flex items-center justify-center">
                <AlertCircle className="w-8 h-8 text-green-600" />
              </div>
              <p className="text-gray-600 font-medium">
                {filter === "active" ? "No active alerts" : "No alerts"}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                {filter === "active" ? "All patients are doing well" : "No alerts have been generated"}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Address Alert Dialog */}
      <Dialog open={addressDialog.open} onOpenChange={(open) => setAddressDialog({ ...addressDialog, open })}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Address Alert</DialogTitle>
            <DialogDescription>
              Add notes about the action taken for this alert
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {addressDialog.alert && (
              <div className="p-3 bg-gray-50 rounded border">
                <p className="text-sm font-semibold text-gray-900">{addressDialog.alert.title}</p>
                <p className="text-sm text-gray-700 mt-1">{addressDialog.alert.message}</p>
                <p className="text-xs text-gray-500 mt-2">
                  Patient: {addressDialog.alert.patientName}
                </p>
              </div>
            )}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-2 block">
                Notes (Optional)
              </label>
              <Textarea
                placeholder="Describe the action taken (e.g., 'Scheduled emergency session', 'Contacted patient', 'Referred to crisis support')..."
                value={addressDialog.notes}
                onChange={(e) => setAddressDialog({ ...addressDialog, notes: e.target.value })}
                rows={4}
                className="w-full"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAddressDialog({ open: false, alert: null, notes: "" })}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddressAlert}
              className="bg-green-600 hover:bg-green-700"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              Mark as Addressed
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
