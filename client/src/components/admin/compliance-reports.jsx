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

export default function ComplianceReports() {
  const [reports, setReports] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/admin/compliance-reports", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setReports(data);
      }
    } catch (error) {
      console.error("Failed to fetch reports:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    try {
      const token = localStorage.getItem("token");
      const res = await fetch("/api/admin/generate-report", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        fetchReports();
      }
    } catch (error) {
      console.error("Failed to generate report:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading reports...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Generate Compliance Report</CardTitle>
          <CardDescription>
            Create audit and compliance documentation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleGenerateReport} className="w-full">
            Generate New Report
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Reports</CardTitle>
          <CardDescription>Compliance and audit reports</CardDescription>
        </CardHeader>
        <CardContent>
          {reports.length > 0 ? (
            <div className="space-y-3">
              {reports.map((report) => (
                <div
                  key={report._id}
                  className="flex items-center justify-between p-4 bg-muted rounded-lg"
                >
                  <div>
                    <p className="font-semibold">{report.title}</p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(report.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <Button size="sm" variant="outline">
                    Download
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No reports generated yet
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Compliance Checklist</CardTitle>
          <CardDescription>System compliance status</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
            <p className="text-sm">Data encryption enabled</p>
          </div>
          <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
            <p className="text-sm">HIPAA compliance active</p>
          </div>
          <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
            <p className="text-sm">Regular backups configured</p>
          </div>
          <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
            <p className="text-sm">Audit logging enabled</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
