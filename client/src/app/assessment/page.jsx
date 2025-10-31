"use client";

import { useAuth } from "@clerk/nextjs";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Card } from "../../components/ui_1/card";
import { Button } from "../../components/ui_1/button";
import { ChevronLeft, CheckCircle } from "lucide-react";
import { Badge } from "../../components/ui_1/badge";

// Import the assessment configurations
const ASSESSMENTS = {
  gad7: {
    id: 'gad7',
    title: 'GAD-7',
    subtitle: 'Generalized Anxiety Disorder Assessment',
    description: 'A 7-question screening tool to assess anxiety levels',
    questions: [
      "Feeling nervous, anxious, or on edge",
      "Not being able to stop or control worrying", 
      "Worrying too much about different things",
      "Trouble relaxing",
      "Being so restless that it's hard to sit still",
      "Becoming easily annoyed or irritable",
      "Feeling afraid as if something awful might happen"
    ],
    answerOptions: [
      { value: 0, label: "Not at all" },
      { value: 1, label: "Several days" },
      { value: 2, label: "More than half the days" },
      { value: 3, label: "Nearly every day" }
    ],
    getSeverity: (score) => {
      if (score >= 15) return { level: "Severe", color: "bg-red-500" };
      if (score >= 10) return { level: "Moderate", color: "bg-orange-500" };
      if (score >= 5) return { level: "Mild", color: "bg-yellow-500" };
      return { level: "Minimal", color: "bg-green-500" };
    }
  },
  phq9: {
    id: 'phq9',
    title: 'PHQ-9',
    subtitle: 'Patient Health Questionnaire',
    description: 'A 9-question screening tool to assess depression levels',
    questions: [
      "Little interest or pleasure in doing things",
      "Feeling down, depressed, or hopeless",
      "Trouble falling or staying asleep, or sleeping too much",
      "Feeling tired or having little energy",
      "Poor appetite or overeating",
      "Feeling bad about yourself or that you are a failure",
      "Trouble concentrating on things",
      "Moving or speaking slowly or being fidgety/restless",
      "Thoughts that you would be better off dead or of hurting yourself"
    ],
    answerOptions: [
      { value: 0, label: "Not at all" },
      { value: 1, label: "Several days" },
      { value: 2, label: "More than half the days" },
      { value: 3, label: "Nearly every day" }
    ],
    getSeverity: (score) => {
      if (score >= 20) return { level: "Severe", color: "bg-red-500" };
      if (score >= 15) return { level: "Moderately Severe", color: "bg-orange-500" };
      if (score >= 10) return { level: "Moderate", color: "bg-yellow-500" };
      if (score >= 5) return { level: "Mild", color: "bg-lime-500" };
      return { level: "None", color: "bg-green-500" };
    }
  }
};

export default function AssessmentPage() {
  const { userId } = useAuth();
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState('form');
  const [gad7Answers, setGad7Answers] = useState(Array(7).fill(null));
  const [phq9Answers, setPhq9Answers] = useState(Array(9).fill(null));
  const [results, setResults] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const calculateScore = (answers) => {
    return answers.reduce((sum, value) => sum + (value || 0), 0);
  };

  const handleSubmit = async () => {
    // Validate that all questions are answered
    const hasUnansweredGad7 = gad7Answers.includes(null);
    const hasUnansweredPhq9 = phq9Answers.includes(null);
    
    if (hasUnansweredGad7 || hasUnansweredPhq9) {
      alert("Please answer all questions before submitting.");
      return;
    }

    setIsSubmitting(true);
    
    try {
      const gad7Score = calculateScore(gad7Answers);
      const phq9Score = calculateScore(phq9Answers);
      
      const assessment = {
        gad7: {
          score: gad7Score,
          severity: ASSESSMENTS.gad7.getSeverity(gad7Score).level,
          answers: gad7Answers.map((answer, index) => ({
            questionId: index + 1,
            answer,
          })),
        },
        phq9: {
          score: phq9Score,
          severity: ASSESSMENTS.phq9.getSeverity(phq9Score).level,
          answers: phq9Answers.map((answer, index) => ({
            questionId: index + 1,
            answer,
          })),
        },
      };

      const response = await fetch("/api/assessments", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(assessment),
      });

      if (!response.ok) {
        throw new Error("Failed to submit assessment");
      }

      // Set results for display
      setResults({
        gad7: {
          ...assessment.gad7,
          severityDetails: ASSESSMENTS.gad7.getSeverity(gad7Score)
        },
        phq9: {
          ...assessment.phq9,
          severityDetails: ASSESSMENTS.phq9.getSeverity(phq9Score)
        }
      });
      
      setCurrentStep('results');
    } catch (error) {
      console.error("Error submitting assessment:", error);
      alert("Failed to submit assessment. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const QuestionSet = ({ assessment, answers, setAnswers }) => (
    <Card className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">{assessment.title}</h2>
        <p className="text-gray-600 text-sm">{assessment.subtitle}</p>
        <p className="text-gray-500 text-sm mt-2">{assessment.description}</p>
      </div>
      
      <div className="space-y-4">
        {assessment.questions.map((question, index) => (
          <div key={index} className="space-y-3 pb-4 border-b border-gray-100 last:border-0">
            <p className="text-gray-700 font-medium">
              {index + 1}. {question}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {assessment.answerOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setAnswers(prev => {
                    const newAnswers = [...prev];
                    newAnswers[index] = option.value;
                    return newAnswers;
                  })}
                  className={`p-3 rounded-lg border-2 transition-all duration-200 text-sm ${
                    answers[index] === option.value
                      ? "border-indigo-500 bg-indigo-50 text-indigo-900"
                      : "border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );

  if (currentStep === 'results') {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center">
                <button
                  onClick={() => router.push("/patient/dashboard")}
                  className="mr-4 p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100"
                >
                  <ChevronLeft className="h-5 w-5" />
                </button>
                <h1 className="text-2xl font-bold text-gray-900">Assessment Results</h1>
              </div>
            </div>
          </div>
        </header>

        {/* Results Content */}
        <main className="max-w-4xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0 space-y-6">
            <Card className="p-6 text-center">
              <div className="flex items-center justify-center mb-4">
                <CheckCircle className="h-8 w-8 text-green-500 mr-2" />
                <h2 className="text-2xl font-bold text-gray-900">Assessment Complete!</h2>
              </div>
              <p className="text-gray-600">Your mental health assessment has been successfully submitted.</p>
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
              {/* GAD-7 Results */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">GAD-7 (Anxiety) Results</h3>
                <div className="text-center mb-4">
                  <div className="text-4xl font-bold text-gray-900 mb-2">{results.gad7.score}</div>
                  <Badge className={`${results.gad7.severityDetails.color} text-white`}>
                    {results.gad7.severity}
                  </Badge>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700">
                    This score indicates {results.gad7.severity.toLowerCase()} anxiety levels based on the GAD-7 assessment.
                  </p>
                </div>
              </Card>

              {/* PHQ-9 Results */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">PHQ-9 (Depression) Results</h3>
                <div className="text-center mb-4">
                  <div className="text-4xl font-bold text-gray-900 mb-2">{results.phq9.score}</div>
                  <Badge className={`${results.phq9.severityDetails.color} text-white`}>
                    {results.phq9.severity}
                  </Badge>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700">
                    This score indicates {results.phq9.severity.toLowerCase()} depression levels based on the PHQ-9 assessment.
                  </p>
                </div>
              </Card>
            </div>

            {/* Recommendations */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h3>
              <div className="space-y-3 text-sm text-gray-700">
                <p>• Share these results with your therapist or healthcare provider</p>
                <p>• Consider scheduling a follow-up appointment if scores indicate moderate to severe levels</p>
                <p>• Continue monitoring your mental health with regular assessments</p>
                <p>• Practice self-care and stress management techniques</p>
              </div>
            </Card>

            {/* Actions */}
            <div className="flex gap-4">
              <Button 
                onClick={() => router.push("/patient/dashboard")}
                className="flex-1"
              >
                Return to Dashboard
              </Button>
              <Button 
                variant="outline" 
                onClick={() => {
                  setCurrentStep('form');
                  setGad7Answers(Array(7).fill(null));
                  setPhq9Answers(Array(9).fill(null));
                  setResults(null);
                }}
                className="flex-1"
              >
                Take Another Assessment
              </Button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <button
                onClick={() => router.push("/patient/dashboard")}
                className="mr-4 p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100"
              >
                <ChevronLeft className="h-5 w-5" />
              </button>
              <h1 className="text-2xl font-bold text-gray-900">Mental Health Assessment</h1>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0 space-y-6">
          {/* Instructions */}
          <Card className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Assessment Instructions</h2>
            <p className="text-gray-700 mb-2">
              Over the last 2 weeks, how often have you been bothered by any of the following problems?
            </p>
            <p className="text-sm text-gray-500">
              Please answer all questions honestly. Your responses will help us better understand your current mental health status.
            </p>
          </Card>

          {/* Assessment Forms */}
          <div className="space-y-6">
            <QuestionSet 
              assessment={ASSESSMENTS.gad7}
              answers={gad7Answers}
              setAnswers={setGad7Answers}
            />
            
            <QuestionSet 
              assessment={ASSESSMENTS.phq9}
              answers={phq9Answers}
              setAnswers={setPhq9Answers}
            />
          </div>

          {/* Submit Button */}
          <Card className="p-6">
            <Button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition-colors duration-200"
            >
              {isSubmitting ? "Submitting Assessment..." : "Submit Assessment"}
            </Button>
          </Card>
        </div>
      </main>
    </div>
  );
}
