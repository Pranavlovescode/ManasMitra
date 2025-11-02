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
  const [gamesData, setGamesData] = useState(null);
  const [gameInsights, setGameInsights] = useState(null);

  const GAME_LABELS = {
    first: "Visual Search",
    second: "Selective Attention",
    third_test: "Sequence Memory",
    fourth: "Verbal Fluency",
    fifth: "Stroop Color Naming",
    sixth: "Cloze Word Completion",
  };

  useEffect(() => {
    fetchAnalytics();
  }, [patientId]);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(
        `/api/therapists/patient/${patientId}/journal-trends`
      );
      if (res.ok) {
        const data = await res.json();
        // Normalize data for charts
        setMoodData(
          (data.moodsOverTime || []).map((d) => ({
            date: d.date,
            score: d.moodScore,
          }))
        );
        setSentimentData(
          (data.sentimentTrend || []).map((d) => ({
            date: d.date,
            score: d.sentimentScore,
          }))
        );
        setRiskData(
          (data.riskTrend || []).map((d) => ({
            date: d.date,
            score: d.riskScore,
          }))
        );
        setEntriesData(
          (data.entriesOverTime || []).map((d) => ({
            date: d.date,
            count: d.count,
          }))
        );
      }

      // Fetch games analytics
      const gamesRes = await fetch(
        `/api/therapists/patient/${patientId}/games`
      );
      if (gamesRes.ok) {
        const gdata = await gamesRes.json();
        setGamesData(gdata);
        setGameInsights(gdata.insights || null);
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
      {gameInsights && (
        <Card>
          <CardHeader>
            <CardTitle>Cognitive Insights</CardTitle>
            <CardDescription>
              Interpreted from recent cognitive game performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.values(gameInsights.domains || {}).map((d) => (
                <div key={d.key} className="rounded-lg border p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium">{d.label}</div>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        d.level === 'high'
                          ? 'bg-emerald-100 text-emerald-700'
                          : d.level === 'average'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-rose-100 text-rose-700'
                      }`}
                    >
                      {d.level}
                    </span>
                  </div>
                  <div className="text-3xl font-semibold">{d.score}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Trend: <span className="font-medium">{d.trend}</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    {d.narrative}
                  </div>
                </div>
              ))}
            </div>
            {(gameInsights.summary || []).length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-medium mb-1">Summary</div>
                <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                  {gameInsights.summary.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Mood Trend</CardTitle>
          <CardDescription>Patient's mood score over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={
                moodData.length > 0 ? moodData : [{ date: "No data", score: 0 }]
              }
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#8884d8"
                name="Mood"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sentiment Trend</CardTitle>
          <CardDescription>
            Overall sentiment from journal analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={
                sentimentData.length > 0
                  ? sentimentData
                  : [{ date: "No data", score: 0 }]
              }
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#82ca9d"
                name="Sentiment"
              />
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
            <LineChart
              data={
                riskData.length > 0 ? riskData : [{ date: "No data", score: 0 }]
              }
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#ff7c7c"
                name="Risk"
              />
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
            <BarChart
              data={
                entriesData.length > 0
                  ? entriesData
                  : [{ date: "No data", count: 0 }]
              }
            >
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

      <Card>
        <CardHeader>
          <CardTitle>Games Performance</CardTitle>
          <CardDescription>
            Aggregated performance from cognitive games
          </CardDescription>
        </CardHeader>
        <CardContent>
          {gamesData?.byGame ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(gamesData.byGame).map(([gameId, info]) => (
                <div key={gameId} className="rounded-lg border p-4">
                  <div className="text-sm text-gray-500 mb-1">
                    {GAME_LABELS[gameId] || gameId}
                  </div>
                  <div className="text-sm">
                    Sessions: <span className="font-medium">{info.count}</span>
                  </div>
                  <div className="text-sm">
                    Avg Score:{" "}
                    <span className="font-medium">
                      {info.avgScore?.toFixed(1)}
                    </span>
                  </div>
                  {typeof info.avgAccuracy === "number" &&
                    info.avgAccuracy > 0 && (
                      <div className="text-sm">
                        Avg Accuracy:{" "}
                        <span className="font-medium">
                          {Math.round(info.avgAccuracy * 100)}%
                        </span>
                      </div>
                    )}
                  {typeof info.avgReactionMs === "number" &&
                    info.avgReactionMs > 0 && (
                      <div className="text-sm">
                        Avg Reaction:{" "}
                        <span className="font-medium">
                          {Math.round(info.avgReactionMs)} ms
                        </span>
                      </div>
                    )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-muted-foreground">No game results yet.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
