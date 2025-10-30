"use client";

import React, { useState } from "react";
import { Card } from "./ui_1/card";
import { Button } from "./ui_1/button";
import { Label } from "./ui_1/label";

const GAD7_QUESTIONS = [
  "Feeling nervous, anxious, or on edge",
  "Not being able to stop or control worrying",
  "Worrying too much about different things",
  "Trouble relaxing",
  "Being so restless that it's hard to sit still",
  "Becoming easily annoyed or irritable",
  "Feeling afraid as if something awful might happen",
];

const PHQ9_QUESTIONS = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself or that you are a failure",
  "Trouble concentrating on things",
  "Moving or speaking slowly or being fidgety/restless",
  "Thoughts that you would be better off dead or of hurting yourself",
];

const ANSWER_OPTIONS = [
  { value: 0, label: "Not at all" },
  { value: 1, label: "Several days" },
  { value: 2, label: "More than half the days" },
  { value: 3, label: "Nearly every day" },
];

const calculateSeverity = (score, type) => {
  if (type === "gad7") {
    if (score >= 15) return "Severe";
    if (score >= 10) return "Moderate";
    if (score >= 5) return "Mild";
    return "Minimal";
  } else {
    if (score >= 20) return "Severe";
    if (score >= 15) return "Moderately Severe";
    if (score >= 10) return "Moderate";
    if (score >= 5) return "Mild";
    return "None";
  }
};

const QuestionnaireForm = ({ onSubmit }) => {
  const [gad7Answers, setGad7Answers] = useState(Array(7).fill(null));
  const [phq9Answers, setPhq9Answers] = useState(Array(9).fill(null));
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (gad7Answers.includes(null) || phq9Answers.includes(null)) {
      setError("Please answer all questions");
      return;
    }

    const gad7Score = gad7Answers.reduce((sum, value) => sum + value, 0);
    const phq9Score = phq9Answers.reduce((sum, value) => sum + value, 0);

    const assessment = {
      gad7: {
        score: gad7Score,
        severity: calculateSeverity(gad7Score, "gad7"),
        answers: gad7Answers.map((answer, index) => ({
          questionId: index + 1,
          answer,
        })),
      },
      phq9: {
        score: phq9Score,
        severity: calculateSeverity(phq9Score, "phq9"),
        answers: phq9Answers.map((answer, index) => ({
          questionId: index + 1,
          answer,
        })),
      },
    };

    onSubmit(assessment);
  };

  const QuestionSet = ({ title, questions, answers, setAnswers }) => (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
      {questions.map((question, index) => (
        <div
          key={index}
          className="space-y-3 pb-4 border-b border-gray-100 last:border-0"
        >
          <Label className="block text-gray-700">
            {index + 1}. {question}
          </Label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {ANSWER_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => {
                  const newAnswers = [...answers];
                  newAnswers[index] = option.value;
                  setAnswers(newAnswers);
                  setError("");
                }}
                className={`p-3 rounded-md border transition-all duration-200 ${
                  answers[index] === option.value
                    ? "bg-indigo-600 text-white border-indigo-600"
                    : "bg-white hover:bg-gray-50 border-gray-200 text-gray-700 hover:border-indigo-300"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <Card className="bg-white shadow-sm p-6">
        <QuestionSet
          title="GAD-7 (Anxiety Assessment)"
          questions={GAD7_QUESTIONS}
          answers={gad7Answers}
          setAnswers={setGad7Answers}
        />
      </Card>

      <Card className="bg-white shadow-sm p-6">
        <QuestionSet
          title="PHQ-9 (Depression Assessment)"
          questions={PHQ9_QUESTIONS}
          answers={phq9Answers}
          setAnswers={setPhq9Answers}
        />
      </Card>

      {error && <div className="text-red-500 text-sm mt-2">{error}</div>}

      <Button
        type="submit"
        className="w-full bg-indigo-600 text-white py-3 px-4 rounded-md hover:bg-indigo-700 transition-colors duration-200 shadow-sm"
      >
        Submit Assessment
      </Button>
    </form>
  );
};

export default QuestionnaireForm;
