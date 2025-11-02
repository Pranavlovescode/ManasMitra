"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../ui/dialog";
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

export default function AssessmentReportModal({ isOpen, onClose, patientId, patientName }) {
  const [loading, setLoading] = useState(true);
  const [assessmentData, setAssessmentData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen && patientId) {
      fetchAssessmentData();
    }
  }, [isOpen, patientId]);

  const fetchAssessmentData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/assessments/patient/${patientId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch assessment data');
      }

      const data = await response.json();
      setAssessmentData(data);
    } catch (err) {
      console.error('Error fetching assessment data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (color) => {
    const colors = {
      green: 'bg-green-100 text-green-800 border-green-300',
      yellow: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      orange: 'bg-orange-100 text-orange-800 border-orange-300',
      red: 'bg-red-100 text-red-800 border-red-300',
      gray: 'bg-gray-100 text-gray-800 border-gray-300'
    };
    return colors[color] || colors.gray;
  };

  const getChartData = () => {
    if (!assessmentData?.assessments) return null;

    const datasets = [];
    const colors = {
      'GAD-7': { border: 'rgb(59, 130, 246)', bg: 'rgba(59, 130, 246, 0.1)' }, // blue
      'PHQ-9': { border: 'rgb(239, 68, 68)', bg: 'rgba(239, 68, 68, 0.1)' }, // red
      'PSS-10': { border: 'rgb(249, 115, 22)', bg: 'rgba(249, 115, 22, 0.1)' }, // orange
      'ISI': { border: 'rgb(168, 85, 247)', bg: 'rgba(168, 85, 247, 0.1)' } // purple
    };

    Object.entries(assessmentData.assessments).forEach(([type, data]) => {
      if (data && data.length > 0) {
        const sortedData = [...data].sort((a, b) => new Date(a.date) - new Date(b.date));
        
        datasets.push({
          label: type,
          data: sortedData.map(item => ({
            x: new Date(item.date),
            y: item.score
          })),
          borderColor: colors[type]?.border || 'rgb(107, 114, 128)',
          backgroundColor: colors[type]?.bg || 'rgba(107, 114, 128, 0.1)',
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6
        });
      }
    });

    return {
      datasets
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            size: 12,
            weight: 'bold'
          },
          padding: 15,
          usePointStyle: true
        }
      },
      title: {
        display: true,
        text: 'Assessment Scores Over Time',
        font: {
          size: 16,
          weight: 'bold'
        },
        padding: 20
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          title: (context) => {
            const date = new Date(context[0].parsed.x);
            return date.toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short', 
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            });
          },
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          displayFormats: {
            day: 'MMM dd'
          }
        },
        title: {
          display: true,
          text: 'Date',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: false
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Score',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  const getMaxScore = (type) => {
    const maxScores = {
      'GAD-7': 21,
      'PHQ-9': 27,
      'PSS-10': 40,
      'ISI': 28
    };
    return maxScores[type] || 100;
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto bg-white">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-gray-900">
            Assessment Report - {patientName}
          </DialogTitle>
          <DialogDescription className="text-gray-700">
            Comprehensive mental health assessment overview
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-4 border-indigo-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading assessment data...</p>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <p className="text-red-600 font-semibold">{error}</p>
            <Button onClick={fetchAssessmentData} className="mt-4">
              Retry
            </Button>
          </div>
        ) : !assessmentData?.summary?.hasData ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📋</div>
            <p className="text-gray-600 text-lg">No assessment data available yet</p>
            <p className="text-gray-500 text-sm mt-2">Patient hasn't completed any assessments</p>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(assessmentData.summary.latestScores).map(([type, data]) => (
                <Card key={type} className="border-2 hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-semibold text-gray-700">{type}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-gray-900">{data.score}</span>
                        <span className="text-sm text-gray-500">/ {getMaxScore(type)}</span>
                      </div>
                      <Badge className={`${getSeverityColor(data.severity.color)} border font-semibold`}>
                        {data.severity.level}
                      </Badge>
                      <p className="text-xs text-gray-600 mt-2">
                        {new Date(data.date).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Chart */}
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-lg font-bold text-gray-900">Score Trends</CardTitle>
                <CardDescription className="text-gray-700">
                  Track assessment scores over time to monitor progress
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  {getChartData() && (
                    <Line data={getChartData()} options={chartOptions} />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Assessment History Details */}
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-lg font-bold text-gray-900">Assessment History</CardTitle>
                <CardDescription className="text-gray-700">
                  Total assessments completed: {assessmentData.summary.totalAssessments}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(assessmentData.assessments).map(([type, data]) => (
                    data && data.length > 0 && (
                      <div key={type} className="p-4 bg-gray-50 rounded-lg border">
                        <h4 className="font-semibold text-gray-900 mb-3">{type}</h4>
                        <div className="space-y-2">
                          {data.slice(0, 5).map((item, idx) => (
                            <div key={idx} className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">
                                {new Date(item.date).toLocaleDateString('en-US', {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                              <span className="font-semibold text-gray-900">
                                Score: {item.score}
                              </span>
                            </div>
                          ))}
                          {data.length > 5 && (
                            <p className="text-xs text-gray-500 mt-2">
                              +{data.length - 5} more assessments
                            </p>
                          )}
                        </div>
                      </div>
                    )
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex justify-end gap-3 pt-4">
              <Button variant="outline" onClick={onClose} className="font-semibold">
                Close
              </Button>
              <Button 
                onClick={() => window.print()} 
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold"
              >
                Print Report
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
