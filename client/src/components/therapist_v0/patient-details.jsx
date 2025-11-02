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
import AssessmentReportModal from "./assessment-report-modal";

export default function PatientDetails({ patientId, therapistId }) {
  // Type annotation removed
  const [patient, setPatient] = useState(null); // Type <any> removed
  const [journals, setJournals] = useState([]); // Type <any[]> removed
  const [moods, setMoods] = useState([]); // Type <any[]> removed
  const [assessments, setAssessments] = useState([]); // Type <any[]> removed
  const [journalTrends, setJournalTrends] = useState(null); // aggregated trends
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [showAssessmentReport, setShowAssessmentReport] = useState(false);

  useEffect(() => {
    fetchPatientData();
  }, [patientId]);

  const fetchPatientData = async () => {
    try {
      const [patientRes, journalsRes, trendsRes, moodsRes, assessmentsRes] =
        await Promise.all([
          fetch(`/api/therapists/patient/${patientId}`),
          // therapist-scoped journals API
          fetch(`/api/therapists/patient/${patientId}/journals`),
          // aggregated trends for journals
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
        // new API returns { items, page, total }
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
            <Button 
              onClick={() => setShowAssessmentReport(true)}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold"
            >
              📊 View Assessment Report
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
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
              {assessments.slice(0, 5).map(
                (
                  assessment // Type :any removed
                ) => (
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
                )
              )}
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
              {journals.slice(0, 3).map(
                (
                  journal // Type :any removed
                ) => (
                  <div key={journal._id} className="p-4 bg-muted rounded-lg">
                    <h3 className="font-semibold">{journal.title}</h3>
                    <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                      {journal.content}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      {new Date(journal.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                )
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Assessment Report Modal */}
      <AssessmentReportModal
        isOpen={showAssessmentReport}
        onClose={() => setShowAssessmentReport(false)}
        patientId={patientId}
        patientName={`${patient?.userId?.firstName} ${patient?.userId?.lastName}`}
      />
    </div>
  );
}
