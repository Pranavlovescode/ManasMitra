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
import { Textarea } from "../ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Input } from "../ui/input";
import { Calendar, TrendingUp, TrendingDown, Activity, BarChart3, Brain } from "lucide-react";
import { LineChart, Line as RechartsLine, BarChart, Bar as RechartsBar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import AssessmentReportModal from "./assessment-report-modal";
import ComprehensiveReportModal from "./comprehensive-report-modal";

export default function PatientDetails({ patientId, therapistId }) {
  const [patient, setPatient] = useState(null);
  const [journals, setJournals] = useState([]);
  const [moods, setMoods] = useState([]);
  const [assessments, setAssessments] = useState([]);
  const [journalTrends, setJournalTrends] = useState(null);
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [showAssessmentReport, setShowAssessmentReport] = useState(false);
  const [showComprehensiveReport, setShowComprehensiveReport] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  useEffect(() => {
    fetchPatientData();
    // Set default date range for analytics
    const end = new Date();
    const start = new Date();
    start.setMonth(start.getMonth() - 6);
    setEndDate(end.toISOString().split('T')[0]);
    setStartDate(start.toISOString().split('T')[0]);
  }, [patientId]);

  useEffect(() => {
    if (activeTab === "details" && startDate && endDate) {
      fetchAnalytics();
    }
  }, [activeTab, startDate, endDate, patientId]);

  const fetchPatientData = async () => {
    try {
      const [patientRes, journalsRes, trendsRes, moodsRes, assessmentsRes] =
        await Promise.all([
          fetch(`/api/therapists/patient/${patientId}`),
          fetch(`/api/therapists/patient/${patientId}/journals`),
          fetch(`/api/therapists/patient/${patientId}/journal-trends`),
          fetch(`/api/therapists/patient/${patientId}/moods`),
          fetch(`/api/therapists/patient/${patientId}/assessments`),
        ]);

      if (patientRes.ok) {
        const data = await patientRes.json();
        setPatient(data);
      }
      if (journalsRes.ok) {
        const j = await journalsRes.json();
        setJournals(Array.isArray(j) ? j : (j.items || []));
      }
      if (trendsRes.ok) setJournalTrends(await trendsRes.json());
      if (moodsRes.ok) setMoods(await moodsRes.json());
      if (assessmentsRes.ok) setAssessments(await assessmentsRes.json());
    } catch (error) {
      console.error("Failed to fetch patient data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    try {
      setAnalyticsLoading(true);
      const res = await fetch(`/api/therapists/patient/${patientId}/journal-trends`);
      if (res.ok) {
        const data = await res.json();
        setAnalyticsData(data);
      }
    } catch (error) {
      console.error("Failed to fetch analytics:", error);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const handleSaveNotes = async () => {
    setIsSaving(true);
    try {
      await fetch(`/api/therapist/notes/${patientId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ notes: clinicalNotes }),
      });
    } catch (error) {
      console.error("Failed to save notes:", error);
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center text-muted-foreground">
        Loading patient data...
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="text-center text-muted-foreground">Patient not found</div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>
                {patient.userId?.firstName} {patient.userId?.lastName}
              </CardTitle>
              <CardDescription>{patient.userId?.email}</CardDescription>
            </div>
            <div className="flex gap-3">
              <Button 
                onClick={() => setShowAssessmentReport(true)}
                variant="outline"
                className="font-semibold"
              >
                📊 Assessment Report
              </Button>
              <Button 
                onClick={() => setShowComprehensiveReport(true)}
                className="bg-linear-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold shadow-lg"
              >
                📋 Comprehensive Report
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Tabbed Interface */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2 h-auto p-1 bg-slate-100 rounded-lg">
          <TabsTrigger value="overview" className="text-base font-semibold py-2">📋 Overview</TabsTrigger>
          <TabsTrigger value="details" className="text-base font-semibold py-2">📊 Details & Analytics</TabsTrigger>
        </TabsList>

        {/* OVERVIEW TAB */}
        <TabsContent value="overview" className="space-y-6">
          <Card>
            <CardContent className="space-y-4 pt-6">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    Total Moods Logged
                  </p>
                  <p className="text-2xl font-bold">{moods.length}</p>
                </div>
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground">Journal Entries</p>
                  <p className="text-2xl font-bold">{journals.length}</p>
                </div>
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    Assessments Completed
                  </p>
                  <p className="text-2xl font-bold">{assessments.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {journalTrends && (
            <Card>
              <CardHeader>
                <CardTitle>Journal Trends</CardTitle>
                <CardDescription>
                  Summary of mood, sentiment, risk, and cognitive patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Mood Direction</p>
                    <p className="text-2xl font-bold capitalize">{journalTrends.directions?.mood || 'unknown'}</p>
                  </div>
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Sentiment Direction</p>
                    <p className="text-2xl font-bold capitalize">{journalTrends.directions?.sentiment || 'unknown'}</p>
                  </div>
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Risk Direction</p>
                    <p className="text-2xl font-bold capitalize">{journalTrends.directions?.risk || 'unknown'}</p>
                  </div>
                </div>
                {journalTrends.topDistortions?.length > 0 && (
                  <div className="mt-4">
                    <p className="text-sm text-muted-foreground mb-1">Top Cognitive Distortions</p>
                    <ul className="list-disc list-inside text-sm">
                      {journalTrends.topDistortions.map((d) => (
                        <li key={d.name}>
                          <span className="capitalize">{d.name}</span> — {d.count}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Clinical Notes</CardTitle>
              <CardDescription>
                Add your observations and treatment notes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Enter clinical notes..."
                value={clinicalNotes}
                onChange={(e) => setClinicalNotes(e.target.value)}
                className="min-h-32"
              />
              <Button onClick={handleSaveNotes} disabled={isSaving}>
                {isSaving ? "Saving..." : "Save Notes"}
              </Button>
            </CardContent>
          </Card>

          {assessments.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Assessment History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {assessments.slice(0, 5).map((assessment) => (
                    <div
                      key={assessment._id}
                      className="flex items-center justify-between p-3 bg-muted rounded-lg"
                    >
                      <div>
                        <p className="font-semibold uppercase">
                          {assessment.type}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(assessment.createdAt).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold">{assessment.score}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {journals.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recent Journal Entries</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {journals.slice(0, 3).map((journal) => (
                    <div key={journal._id} className="p-4 bg-muted rounded-lg">
                      <h3 className="font-semibold">{journal.title}</h3>
                      <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                        {journal.content}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        {new Date(journal.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* DETAILS TAB WITH TIMELINE SELECTION */}
        <TabsContent value="details" className="space-y-6">
          <Card className="border-2 border-indigo-200 bg-linear-to-r from-indigo-50 to-purple-50">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-indigo-600" />
                  <span className="font-semibold text-gray-900">Date Range:</span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">From:</label>
                  <Input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-40"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">To:</label>
                  <Input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-40"
                  />
                </div>
                <Button 
                  onClick={() => {
                    const end = new Date();
                    const start = new Date();
                    start.setMonth(start.getMonth() - 6);
                    setEndDate(end.toISOString().split('T')[0]);
                    setStartDate(start.toISOString().split('T')[0]);
                  }}
                  variant="outline"
                  size="sm"
                >
                  Reset to Last 6 Months
                </Button>
              </div>
            </CardContent>
          </Card>

          {analyticsLoading ? (
            <Card>
              <CardContent className="py-12 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-4 border-indigo-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading analytics...</p>
              </CardContent>
            </Card>
          ) : analyticsData ? (
            <>
              {/* Mood Trend Chart */}
              {analyticsData.moodsOverTime && analyticsData.moodsOverTime.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-blue-600" />
                      Mood Trend
                    </CardTitle>
                    <CardDescription>Patient's mood score over the selected period</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={analyticsData.moodsOverTime}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <RechartsLine
                          type="monotone"
                          dataKey="moodScore"
                          stroke="#3b82f6"
                          name="Mood Score"
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Sentiment Trend Chart */}
              {analyticsData.sentimentTrend && analyticsData.sentimentTrend.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-green-600" />
                      Sentiment Trend
                    </CardTitle>
                    <CardDescription>Overall sentiment from journal analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={analyticsData.sentimentTrend}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <RechartsLine
                          type="monotone"
                          dataKey="sentimentScore"
                          stroke="#82ca9d"
                          name="Sentiment"
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Risk Trend Chart */}
              {analyticsData.riskTrend && analyticsData.riskTrend.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingDown className="w-5 h-5 text-red-600" />
                      Risk Trend
                    </CardTitle>
                    <CardDescription>Risk score from journal analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={analyticsData.riskTrend}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <RechartsLine
                          type="monotone"
                          dataKey="riskScore"
                          stroke="#ff7c7c"
                          name="Risk"
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Entry Frequency Chart */}
              {analyticsData.entriesOverTime && analyticsData.entriesOverTime.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-purple-600" />
                      Entry Frequency
                    </CardTitle>
                    <CardDescription>Number of journal entries per day</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={analyticsData.entriesOverTime}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <RechartsBar dataKey="count" fill="#8884d8" name="Entries" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-gray-600">No analytics data available for the selected date range</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Assessment Report Modal */}
      <AssessmentReportModal
        isOpen={showAssessmentReport}
        onClose={() => setShowAssessmentReport(false)}
        patientId={patientId}
        patientName={`${patient?.userId?.firstName} ${patient?.userId?.lastName}`}
      />

      {/* Comprehensive Report Modal */}
      <ComprehensiveReportModal
        isOpen={showComprehensiveReport}
        onClose={() => setShowComprehensiveReport(false)}
        patientId={patientId}
        patientName={`${patient?.userId?.firstName} ${patient?.userId?.lastName}`}
      />
    </div>
  );
}
