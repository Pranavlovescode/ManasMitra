"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function AnalyticsDashboard({ patientId }) {
  // Type annotation removed
  const [moodData, setMoodData] = useState([]); // Type <any[]> removed
  const [sentimentData, setSentimentData] = useState([]); // Type <any[]> removed
  const [riskData, setRiskData] = useState([]); // Type <any[]> removed
  const [entriesData, setEntriesData] = useState([]); // Type <any[]> removed
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, [patientId]);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`/api/therapists/patient/${patientId}/journal-trends`);
      if (res.ok) {
        const data = await res.json();
        // Normalize data for charts
        setMoodData((data.moodsOverTime || []).map(d => ({ date: d.date, score: d.moodScore })));
        setSentimentData((data.sentimentTrend || []).map(d => ({ date: d.date, score: d.sentimentScore })));
        setRiskData((data.riskTrend || []).map(d => ({ date: d.date, score: d.riskScore })));
        setEntriesData((data.entriesOverTime || []).map(d => ({ date: d.date, count: d.count })));
      }
    } catch (error) {
      console.error("Failed to fetch analytics:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading analytics...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Mood Trend</CardTitle>
          <CardDescription>Patient's mood score over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={moodData.length > 0 ? moodData : [{ date: "No data", score: 0 }]}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#8884d8" name="Mood" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sentiment Trend</CardTitle>
          <CardDescription>Overall sentiment from journal analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={sentimentData.length > 0 ? sentimentData : [{ date: "No data", score: 0 }]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#82ca9d" name="Sentiment" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Risk Trend</CardTitle>
          <CardDescription>Risk score from journal analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={riskData.length > 0 ? riskData : [{ date: "No data", score: 0 }]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#ff7c7c" name="Risk" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Entry Frequency</CardTitle>
          <CardDescription>Number of journal entries per day</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={entriesData.length > 0 ? entriesData : [{ date: 'No data', count: 0 }] }>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#8884d8" name="Entries" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
