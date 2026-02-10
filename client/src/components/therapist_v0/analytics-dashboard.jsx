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
import { getGameById, getGameName, getCognitiveDomain, getDomainColors } from '@/lib/game-config';
import { Gamepad2, TrendingUp, TrendingDown, Brain } from 'lucide-react';
import { Badge } from "../ui/badge";

export default function AnalyticsDashboard({ patientId }) {
  // Type annotation removed
  const [moodData, setMoodData] = useState([]); // Type <any[]> removed
  const [sentimentData, setSentimentData] = useState([]); // Type <any[]> removed
  const [riskData, setRiskData] = useState([]); // Type <any[]> removed
  const [entriesData, setEntriesData] = useState([]); // Type <any[]> removed
  const [isLoading, setIsLoading] = useState(true);
  const [gamesData, setGamesData] = useState(null);
  const [gameInsights, setGameInsights] = useState(null);



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

      <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gamepad2 className="w-6 h-6 text-green-600" />
            Cognitive Games Performance
          </CardTitle>
          <CardDescription>
            Detailed performance metrics from cognitive games
          </CardDescription>
        </CardHeader>
        <CardContent>
          {gamesData?.byGame ? (
            <>
              {/* Summary Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <Card className="bg-white">
                  <CardContent className="pt-6 text-center">
                    <p className="text-sm text-gray-600 mb-1">Total Sessions</p>
                    <p className="text-3xl font-bold text-green-700">
                      {Object.values(gamesData.byGame).reduce((sum, g) => sum + g.count, 0)}
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardContent className="pt-6 text-center">
                    <p className="text-sm text-gray-600 mb-1">Games Played</p>
                    <p className="text-3xl font-bold text-blue-700">
                      {Object.keys(gamesData.byGame).length}
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardContent className="pt-6 text-center">
                    <p className="text-sm text-gray-600 mb-1">Avg Performance</p>
                    <p className="text-3xl font-bold text-purple-700">
                      {(
                        Object.values(gamesData.byGame).reduce((sum, g) => sum + (g.avgScore || 0), 0) /
                        Object.keys(gamesData.byGame).length
                      ).toFixed(1)}
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Individual Game Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(gamesData.byGame).map(([gameId, info]) => {
                  const game = getGameById(gameId);
                  const domainColors = getDomainColors(getCognitiveDomain(gameId));
                  return (
                    <div key={gameId} className={`p-4 ${domainColors.bg} border-2 ${domainColors.border} rounded-lg hover:shadow-lg transition-all`}>
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">{game?.emoji || '🎮'}</span>
                          <div>
                            <div className="text-sm font-semibold text-gray-900">
                              {getGameName(gameId)}
                            </div>
                            <div className="text-xs text-gray-600">
                              {game?.cognitiveDomain || 'Cognitive'}
                            </div>
                          </div>
                        </div>
                        <Badge className="bg-white text-gray-900 border border-gray-300">
                          {info.count}x
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-600">Avg Score:</span>
                          <span className="text-lg font-bold text-gray-900">
                            {info.avgScore?.toFixed(1) || 'N/A'}
                          </span>
                        </div>
                        
                        {typeof info.avgAccuracy === "number" && info.avgAccuracy > 0 && (
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-600">Accuracy:</span>
                            <span className="text-sm font-semibold text-gray-900">
                              {Math.round(info.avgAccuracy * 100)}%
                            </span>
                          </div>
                        )}
                        
                        {typeof info.avgReactionMs === "number" && info.avgReactionMs > 0 && (
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-600">Reaction:</span>
                            <span className="text-sm font-semibold text-gray-900">
                              {Math.round(info.avgReactionMs)}ms
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Gamepad2 className="w-12 h-12 mx-auto mb-2 text-gray-400" />
              <p>No game results yet.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
