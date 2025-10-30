"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";

const PHQ9_QUESTIONS = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself",
  "Trouble concentrating on things",
  "Moving or speaking so slowly that others have noticed",
  "Thoughts that you would be better off dead",
];

const GAD7_QUESTIONS = [
  "Feeling nervous, anxious or on edge",
  "Not being able to stop or control worrying",
  "Worrying too much about different things",
  "Trouble relaxing",
  "Being so restless that it is hard to sit still",
  "Becoming easily annoyed or irritable",
  "Feeling afraid as if something awful might happen",
];

export default function AssessmentModule({ userId }) {
  // Type annotation removed
  const [selectedAssessment, setSelectedAssessment] = useState(null); // Type annotation removed
  const [responses, setResponses] = useState([]); // Type annotation removed
  const [score, setScore] = useState(null); // Type annotation removed
  const [isLoading, setIsLoading] = useState(false);

  const questions =
    selectedAssessment === "phq9" ? PHQ9_QUESTIONS : GAD7_QUESTIONS;

  const handleResponseChange = (index, value) => {
    // Type annotations removed
    const newResponses = [...responses];
    newResponses[index] = value;
    setResponses(newResponses);
  };

  const handleSubmit = async () => {
    if (responses.length !== questions.length) {
      alert("Please answer all questions");
      return;
    }

    const totalScore = responses.reduce((a, b) => a + b, 0);
    setScore(totalScore);

    setIsLoading(true);

    try {
      const token = localStorage.getItem("token");
      await fetch("/api/assessment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          userId,
          type: selectedAssessment,
          score: totalScore,
          responses,
        }),
      });
    } catch (error) {
      console.error("Failed to save assessment:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedAssessment(null);
    setResponses([]);
    setScore(null);
  };

  if (score !== null) {
    const interpretation =
      selectedAssessment === "phq9"
        ? score < 5
          ? "Minimal depression"
          : score < 10
          ? "Mild depression"
          : score < 15
          ? "Moderate depression"
          : score < 20
          ? "Moderately severe depression"
          : "Severe depression"
        : score < 5
        ? "Minimal anxiety"
        : score < 10
        ? "Mild anxiety"
        : score < 15
        ? "Moderate anxiety"
        : score < 21
        ? "Severe anxiety"
        : "Very severe anxiety";

    return (
      <Card>
        <CardHeader>
          <CardTitle>Assessment Results</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center">
            <div className="text-5xl font-bold text-primary mb-2">{score}</div>
            <p className="text-lg font-semibold">{interpretation}</p>
            <p className="text-sm text-muted-foreground mt-2">
              {selectedAssessment === "phq9" ? "PHQ-9 Score" : "GAD-7 Score"}
            </p>
          </div>
          <p className="text-sm text-muted-foreground">
            Consider discussing these results with your therapist to develop a
            personalized treatment plan.
          </p>
          <Button onClick={handleReset} className="w-full">
            Take Another Assessment
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!selectedAssessment) {
    return (
      <div className="grid md:grid-cols-2 gap-4">
        <Card
          className="cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => {
            setSelectedAssessment("phq9");
            setResponses(new Array(9).fill(0));
          }}
        >
          <CardHeader>
            <CardTitle>PHQ-9</CardTitle>
            <CardDescription>Patient Health Questionnaire</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Assess your depression levels with this 9-question screening tool.
            </p>
          </CardContent>
        </Card>

        <Card
          className="cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => {
            setSelectedAssessment("gad7");
            setResponses(new Array(7).fill(0));
          }}
        >
          <CardHeader>
            <CardTitle>GAD-7</CardTitle>
            <CardDescription>Generalized Anxiety Disorder</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Assess your anxiety levels with this 7-question screening tool.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {selectedAssessment === "phq9" ? "PHQ-9" : "GAD-7"} Assessment
        </CardTitle>
        <CardDescription>
          Rate each statement from 0 (Not at all) to 3 (Nearly every day)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {questions.map((question, index) => (
          <div key={index} className="space-y-2">
            <p className="font-medium">{question}</p>
            <div className="flex gap-2">
              {[0, 1, 2, 3].map((value) => (
                <button
                  key={value}
                  onClick={() => handleResponseChange(index, value)}
                  className={`flex-1 py-2 rounded-lg transition-all ${
                    responses[index] === value
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                >
                  {value === 0
                    ? "Not"
                    : value === 1
                    ? "Several"
                    : value === 2
                    ? "More"
                    : "Nearly"}
                </button>
              ))}
            </div>
          </div>
        ))}

        <div className="flex gap-2">
          <Button
            onClick={handleReset}
            variant="outline"
            className="flex-1 bg-transparent"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isLoading}
            className="flex-1"
          >
            {isLoading ? "Submitting..." : "Submit Assessment"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
