"use client";

import { useState, useEffect } from "react";
import { Button } from "../ui_1/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui_1/card";
import { Badge } from "../ui_1/badge";
import { ChevronLeft, CheckCircle, Clock, AlertCircle } from "lucide-react";

// Assessment configurations
const ASSESSMENTS = {
  gad7: {
    id: 'gad7',
    title: 'GAD-7',
    subtitle: 'Generalized Anxiety Disorder Assessment',
    description: 'A 7-question screening tool to assess anxiety levels',
    icon: '🧠',
    color: 'from-blue-500 to-purple-600',
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
    icon: '💭',
    color: 'from-green-500 to-teal-600',
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
  },
  stress: {
    id: 'stress',
    title: 'PSS-10',
    subtitle: 'Perceived Stress Scale',
    description: 'A 10-question assessment to measure stress levels',
    icon: '⚡',
    color: 'from-orange-500 to-red-600',
    questions: [
      "How often have you been upset because of something that happened unexpectedly?",
      "How often have you felt that you were unable to control the important things in your life?",
      "How often have you felt nervous and stressed?",
      "How often have you felt confident about your ability to handle your personal problems?",
      "How often have you felt that things were going your way?",
      "How often have you found that you could not cope with all the things that you had to do?",
      "How often have you been able to control irritations in your life?",
      "How often have you felt that you were on top of things?",
      "How often have you been angered because of things that happened that were outside of your control?",
      "How often have you felt difficulties were piling up so high that you could not overcome them?"
    ],
    answerOptions: [
      { value: 0, label: "Never" },
      { value: 1, label: "Almost Never" },
      { value: 2, label: "Sometimes" },
      { value: 3, label: "Fairly Often" },
      { value: 4, label: "Very Often" }
    ],
    // Questions 4, 5, 7, 8 are reverse scored
    reverseScored: [3, 4, 6, 7],
    getSeverity: (score) => {
      if (score >= 27) return { level: "High Stress", color: "bg-red-500" };
      if (score >= 14) return { level: "Moderate Stress", color: "bg-yellow-500" };
      return { level: "Low Stress", color: "bg-green-500" };
    }
  },
  sleep: {
    id: 'sleep',
    title: 'ISI',
    subtitle: 'Insomnia Severity Index',
    description: 'A 7-question assessment to evaluate sleep quality',
    icon: '😴',
    color: 'from-indigo-500 to-blue-600',
    questions: [
      "Rate the current severity of your insomnia problem: Difficulty falling asleep",
      "Rate the current severity of your insomnia problem: Difficulty staying asleep", 
      "Rate the current severity of your insomnia problem: Problem waking up too early",
      "How satisfied/dissatisfied are you with your current sleep pattern?",
      "To what extent do you consider your sleep problem to interfere with your daily functioning?",
      "How noticeable to others do you think your sleeping problem is in terms of impairing your quality of life?",
      "How worried/distressed are you about your current sleep problem?"
    ],
    answerOptions: [
      { value: 0, label: "None" },
      { value: 1, label: "Mild" },
      { value: 2, label: "Moderate" },
      { value: 3, label: "Severe" },
      { value: 4, label: "Very Severe" }
    ],
    getSeverity: (score) => {
      if (score >= 22) return { level: "Severe Insomnia", color: "bg-red-500" };
      if (score >= 15) return { level: "Moderate Insomnia", color: "bg-orange-500" };
      if (score >= 8) return { level: "Mild Insomnia", color: "bg-yellow-500" };
      return { level: "No Insomnia", color: "bg-green-500" };
    }
  }
};

export default function MultiAssessmentModule({ userId }) {
  const [currentStep, setCurrentStep] = useState('list'); // 'list', 'taking', 'result'
  const [selectedAssessment, setSelectedAssessment] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [completedAssessments, setCompletedAssessments] = useState([]);

  // Load completed assessments on component mount
  useEffect(() => {
    const loadCompletedAssessments = async () => {
      try {
        const response = await fetch('/api/assessments');
        if (response.ok) {
          const data = await response.json();
          console.log('📊 Loaded completed assessments:', data);
          console.log('📊 Number of assessments:', data.length);
          setCompletedAssessments(data);
        }
      } catch (error) {
        console.error('Error loading assessments:', error);
      }
    };

    if (userId) {
      loadCompletedAssessments();
    }
  }, [userId]);

  const startAssessment = (assessmentId) => {
    const assessment = ASSESSMENTS[assessmentId];
    setSelectedAssessment(assessment);
    setCurrentQuestion(0);
    setAnswers(new Array(assessment.questions.length).fill(null));
    setCurrentStep('taking');
  };

  const handleAnswer = (value) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = value;
    setAnswers(newAnswers);
  };

  const nextQuestion = () => {
    if (currentQuestion < selectedAssessment.questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      calculateAndSubmitResult();
    }
  };

  const prevQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const calculateAndSubmitResult = async () => {
    setIsSubmitting(true);
    
    try {
      // Calculate score
      let score = 0;
      answers.forEach((answer, index) => {
        if (selectedAssessment.reverseScored && selectedAssessment.reverseScored.includes(index)) {
          // Reverse score for specific questions
          const maxValue = Math.max(...selectedAssessment.answerOptions.map(opt => opt.value));
          score += maxValue - answer;
        } else {
          score += answer;
        }
      });

      const severity = selectedAssessment.getSeverity(score);

      // Prepare assessment data
      const assessmentData = {
        [selectedAssessment.id]: {
          score,
          severity: severity.level,
          answers: answers.map((answer, index) => ({
            questionId: index + 1,
            answer
          }))
        }
      };

      // Submit to API
      const response = await fetch('/api/assessments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(assessmentData)
      });

      if (response.ok) {
        setAssessmentResult({
          ...selectedAssessment,
          score,
          severity
        });
        setCurrentStep('result');
        
        // Reload completed assessments
        const updatedResponse = await fetch('/api/assessments');
        if (updatedResponse.ok) {
          const data = await updatedResponse.json();
          setCompletedAssessments(data);
        }
      } else {
        throw new Error('Failed to submit assessment');
      }
    } catch (error) {
      console.error('Error submitting assessment:', error);
      alert('Failed to submit assessment. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetToList = () => {
    setCurrentStep('list');
    setSelectedAssessment(null);
    setCurrentQuestion(0);
    setAnswers([]);
    setAssessmentResult(null);
  };

  const getAssessmentStatus = (assessmentId) => {
    // Find the LATEST assessment for this specific type
    // completedAssessments is sorted by date DESC, so first match is most recent
    const completed = completedAssessments.find(a => a[assessmentId]);
    console.log(`🔍 Checking status for ${assessmentId}:`, completed ? 'Found' : 'Not found');
    if (completed) {
      const data = completed[assessmentId];
      console.log(`   Score: ${data.score}, Severity: ${data.severity}, Date: ${completed.date}`);
      return {
        completed: true,
        date: new Date(completed.date).toLocaleDateString(),
        score: data.score,
        severity: data.severity
      };
    }
    return { completed: false };
  };

  // Assessment List View
  if (currentStep === 'list') {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Mental Health Assessments</h2>
          <p className="text-gray-600">Choose an assessment to evaluate your mental well-being</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.values(ASSESSMENTS).map((assessment) => {
            const status = getAssessmentStatus(assessment.id);
            return (
              <Card 
                key={assessment.id}
                className="hover:shadow-lg transition-all duration-300 cursor-pointer group"
                onClick={() => startAssessment(assessment.id)}
              >
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-full bg-linear-to-r ${assessment.color} flex items-center justify-center text-2xl shadow-lg`}>
                      {assessment.icon}
                    </div>
                    {status.completed && (
                      <Badge className="bg-green-100 text-green-800 border-green-200">
                        <CheckCircle className="w-3 h-3 mr-1" />
                        Completed
                      </Badge>
                    )}
                  </div>
                  <CardTitle className="text-xl group-hover:text-indigo-600 transition-colors">
                    {assessment.title}
                  </CardTitle>
                  <CardDescription className="text-sm font-medium text-gray-600">
                    {assessment.subtitle}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 mb-4">{assessment.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">
                      {assessment.questions.length} questions
                    </span>
                    {status.completed && (
                      <div className="text-right">
                        <div className="text-xs text-gray-400">Last taken: {status.date}</div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">Score: {status.score}</span>
                          <Badge className={`text-xs text-white ${status.severity === 'Severe' || status.severity === 'High Stress' || status.severity === 'Severe Insomnia' ? 'bg-red-500' : 
                            status.severity === 'Moderate' || status.severity === 'Moderately Severe' || status.severity === 'Moderate Stress' || status.severity === 'Moderate Insomnia' ? 'bg-orange-500' :
                            status.severity === 'Mild' || status.severity === 'Mild Insomnia' ? 'bg-yellow-500' : 'bg-green-500'}`}>
                            {status.severity}
                          </Badge>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    );
  }

  // Assessment Taking View
  if (currentStep === 'taking') {
    const progress = ((currentQuestion + 1) / selectedAssessment.questions.length) * 100;
    
    return (
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={resetToList}
            className="flex items-center gap-2"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to Assessments
          </Button>
          <div>
            <h2 className="text-xl font-bold text-gray-900">{selectedAssessment.title}</h2>
            <p className="text-sm text-gray-600">{selectedAssessment.subtitle}</p>
          </div>
        </div>

        {/* Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Question {currentQuestion + 1} of {selectedAssessment.questions.length}</span>
            <span className="text-gray-600">{Math.round(progress)}% complete</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-linear-to-r from-indigo-500 to-purple-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Question Card */}
        <Card className="p-8">
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                {selectedAssessment.questions[currentQuestion]}
              </h3>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {selectedAssessment.answerOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handleAnswer(option.value)}
                  className={`p-4 rounded-lg border-2 transition-all duration-200 text-left ${
                    answers[currentQuestion] === option.value
                      ? "border-indigo-500 bg-indigo-50 text-indigo-900"
                      : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      answers[currentQuestion] === option.value
                        ? "border-indigo-500 bg-indigo-500"
                        : "border-gray-300"
                    }`}>
                      {answers[currentQuestion] === option.value && (
                        <div className="w-full h-full rounded-full bg-white scale-50" />
                      )}
                    </div>
                    <span className="font-medium">{option.label}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </Card>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button 
            variant="outline" 
            onClick={prevQuestion}
            disabled={currentQuestion === 0}
          >
            Previous
          </Button>
          <Button 
            onClick={nextQuestion}
            disabled={answers[currentQuestion] === null}
            className={currentQuestion === selectedAssessment.questions.length - 1 ? 
              "bg-green-600 hover:bg-green-700" : ""}
          >
            {currentQuestion === selectedAssessment.questions.length - 1 ? 
              (isSubmitting ? "Submitting..." : "Submit Assessment") : "Next"}
          </Button>
        </div>
      </div>
    );
  }

  // Results View
  if (currentStep === 'result') {
    return (
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <div className={`w-16 h-16 rounded-full bg-linear-to-r ${assessmentResult.color} flex items-center justify-center text-3xl mx-auto mb-4 shadow-lg`}>
            {assessmentResult.icon}
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Assessment Complete!</h2>
          <p className="text-gray-600">{assessmentResult.title} - {assessmentResult.subtitle}</p>
        </div>

        {/* Results Card */}
        <Card className="p-8 text-center">
          <div className="space-y-6">
            <div>
              <div className="text-5xl font-bold text-gray-900 mb-2">{assessmentResult.score}</div>
              <div className="flex items-center justify-center gap-2">
                <Badge className={`${assessmentResult.severity.color} text-white text-lg px-4 py-2`}>
                  {assessmentResult.severity.level}
                </Badge>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-700">
                <strong>What this means:</strong> This score indicates {assessmentResult.severity.level.toLowerCase()} levels 
                based on the {assessmentResult.title} assessment. Consider discussing these results with a mental health 
                professional for personalized guidance and treatment options.
              </p>
            </div>

            {assessmentResult.severity.level !== 'Minimal' && assessmentResult.severity.level !== 'None' && assessmentResult.severity.level !== 'Low Stress' && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center gap-2 text-blue-800 mb-2">
                  <AlertCircle className="w-4 h-4" />
                  <span className="font-medium">Recommended Actions</span>
                </div>
                <ul className="text-sm text-blue-700 text-left space-y-1">
                  <li>• Schedule a consultation with a mental health professional</li>
                  <li>• Continue regular self-assessments to track progress</li>
                  <li>• Practice self-care and stress management techniques</li>
                  <li>• Consider therapy or counseling options</li>
                </ul>
              </div>
            )}
          </div>
        </Card>

        {/* Actions */}
        <div className="flex gap-4">
          <Button 
            variant="outline" 
            onClick={resetToList}
            className="flex-1"
          >
            Take Another Assessment
          </Button>
          <Button 
            onClick={() => window.print()}
            className="flex-1"
          >
            Save Results
          </Button>
        </div>
      </div>
    );
  }

  return null;
}