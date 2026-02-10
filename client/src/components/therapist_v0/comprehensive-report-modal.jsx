"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../ui/dialog";
import { Input } from "../ui/input";
import { AlertCircle, TrendingUp, TrendingDown, Minus, Activity, Brain, MessageSquare, Gamepad2, BookOpen, Heart, Calendar, BarChart3 } from "lucide-react";
import { Line, Bar } from 'react-chartjs-2';
import { LineChart, Line as RechartsLine, BarChart, Bar as RechartsBar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { format, parseISO, isValid } from 'date-fns';
import { getGameById, getGameName, getCognitiveDomain, getDomainColors, getGamesByDomain } from '@/lib/game-config';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend,
  TimeScale
);

export default function ComprehensiveReportModal({ isOpen, onClose, patientId, patientName }) {
  const [loading, setLoading] = useState(true);
  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [filteredData, setFilteredData] = useState(null);

  useEffect(() => {
    if (isOpen && patientId) {
      fetchReportData();
      // Set default date range to last 6 months
      const end = new Date();
      const start = new Date();
      start.setMonth(start.getMonth() - 6);
      setEndDate(format(end, 'yyyy-MM-dd'));
      setStartDate(format(start, 'yyyy-MM-dd'));
    }
  }, [isOpen, patientId]);

  useEffect(() => {
    if (reportData && startDate && endDate) {
      filterDataByDateRange();
    }
  }, [startDate, endDate, reportData]);

  const fetchReportData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/therapists/patient/${patientId}/comprehensive-report`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error Response:', errorText);
        throw new Error(`Failed to fetch comprehensive report (${response.status})`);
      }

      const data = await response.json();
      console.log('Comprehensive Report Data:', data);
      setReportData(data);
    } catch (err) {
      console.error('Error fetching comprehensive report:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filterDataByDateRange = () => {
    if (!reportData || !startDate || !endDate) {
      setFilteredData(reportData);
      return;
    }

    const start = new Date(startDate);
    const end = new Date(endDate);
    end.setHours(23, 59, 59, 999); // Include entire end date

    const filtered = JSON.parse(JSON.stringify(reportData)); // Deep clone

    // Filter assessments
    if (filtered.assessments?.summary) {
      Object.keys(filtered.assessments.summary).forEach(type => {
        const data = filtered.assessments.summary[type];
        if (data.scores) {
          data.scores = data.scores.filter(item => {
            const itemDate = new Date(item.date);
            return itemDate >= start && itemDate <= end;
          });
          // Update latest if scores exist
          if (data.scores.length > 0) {
            data.latest = {
              ...data.scores[0],
              severity: getSeverityLevel(type, data.scores[0].score)
            };
          } else {
            data.latest = null;
          }
        }
      });
    }

    // Filter journals
    if (filtered.journals) {
      if (filtered.journals.emotionTrends) {
        filtered.journals.emotionTrends = filtered.journals.emotionTrends.filter(item => {
          const itemDate = new Date(item.date);
          return itemDate >= start && itemDate <= end;
        });
      }
      if (filtered.journals.riskIndicators) {
        filtered.journals.riskIndicators = filtered.journals.riskIndicators.filter(item => {
          const itemDate = new Date(item.date);
          return itemDate >= start && itemDate <= end;
        });
      }
    }

    setFilteredData(filtered);
  };

  const safeParseDate = (dateValue) => {
    if (!dateValue) return null;
    
    try {
      // If it's already a Date object
      if (dateValue instanceof Date) {
        return isValid(dateValue) ? dateValue : null;
      }
      
      // If it's a string
      if (typeof dateValue === 'string') {
        const parsed = parseISO(dateValue);
        return isValid(parsed) ? parsed : null;
      }
      
      // Try to create a Date object
      const date = new Date(dateValue);
      return isValid(date) ? date : null;
    } catch (e) {
      console.warn('Date parsing error:', e, dateValue);
      return null;
    }
  };

  const getSeverityLevel = (type, score) => {
    const severityRanges = {
      'GAD-7': [
        { min: 0, max: 4, level: 'Minimal', color: 'green' },
        { min: 5, max: 9, level: 'Mild', color: 'yellow' },
        { min: 10, max: 14, level: 'Moderate', color: 'orange' },
        { min: 15, max: 21, level: 'Severe', color: 'red' }
      ],
      'PHQ-9': [
        { min: 0, max: 4, level: 'None-Minimal', color: 'green' },
        { min: 5, max: 9, level: 'Mild', color: 'yellow' },
        { min: 10, max: 14, level: 'Moderate', color: 'orange' },
        { min: 15, max: 19, level: 'Moderately Severe', color: 'orange' },
        { min: 20, max: 27, level: 'Severe', color: 'red' }
      ],
      'PSS-10': [
        { min: 0, max: 13, level: 'Low Stress', color: 'green' },
        { min: 14, max: 26, level: 'Moderate Stress', color: 'yellow' },
        { min: 27, max: 40, level: 'High Stress', color: 'red' }
      ],
      'ISI': [
        { min: 0, max: 7, level: 'No Insomnia', color: 'green' },
        { min: 8, max: 14, level: 'Subthreshold', color: 'yellow' },
        { min: 15, max: 21, level: 'Moderate', color: 'orange' },
        { min: 22, max: 28, level: 'Severe', color: 'red' }
      ]
    };

    const ranges = severityRanges[type] || [];
    for (const range of ranges) {
      if (score >= range.min && score <= range.max) {
        return { level: range.level, color: range.color };
      }
    }
    return { level: 'Unknown', color: 'gray' };
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

  const getTrendIcon = (direction) => {
    if (direction === 'improving') return <TrendingDown className="w-4 h-4 text-green-600" />;
    if (direction === 'worsening') return <TrendingUp className="w-4 h-4 text-red-600" />;
    return <Minus className="w-4 h-4 text-gray-600" />;
  };

  const getAlertColor = (priority) => {
    if (priority === 'urgent') return 'bg-red-50 border-red-300 text-red-900';
    if (priority === 'high') return 'bg-orange-50 border-orange-300 text-orange-900';
    return 'bg-yellow-50 border-yellow-300 text-yellow-900';
  };

  // Generate Assessment Timeline Chart
  const getAssessmentChartData = () => {
    const data = filteredData || reportData;
    if (!data?.assessments?.summary) return null;

    const datasets = [];
    const colors = {
      'GAD-7': { border: 'rgb(59, 130, 246)', bg: 'rgba(59, 130, 246, 0.1)' },
      'PHQ-9': { border: 'rgb(239, 68, 68)', bg: 'rgba(239, 68, 68, 0.1)' },
      'PSS-10': { border: 'rgb(249, 115, 22)', bg: 'rgba(249, 115, 22, 0.1)' },
      'ISI': { border: 'rgb(168, 85, 247)', bg: 'rgba(168, 85, 247, 0.1)' }
    };

    Object.entries(data.assessments.summary).forEach(([type, typeData]) => {
      if (typeData.scores && typeData.scores.length > 0) {
        const sortedData = [...typeData.scores].sort((a, b) => {
          const dateA = safeParseDate(a.date);
          const dateB = safeParseDate(b.date);
          if (!dateA || !dateB) return 0;
          return dateA - dateB;
        });

        datasets.push({
          label: type,
          data: sortedData.map(item => {
            const parsedDate = safeParseDate(item.date);
            return {
              x: parsedDate || new Date(),
              y: item.score
            };
          }).filter(item => item.x),
          borderColor: colors[type]?.border || 'rgb(107, 114, 128)',
          backgroundColor: colors[type]?.bg || 'rgba(107, 114, 128, 0.1)',
          tension: 0.3,
          borderWidth: 3,
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: false
        });
      }
    });

    return datasets.length > 0 ? { datasets } : null;
  };

  // Generate Mood Timeline Chart
  const getMoodChartData = () => {
    const data = filteredData || reportData;
    if (!data?.moods || !data.moods.totalEntries) return null;

    // Create a mood intensity chart (placeholder - you'll need actual mood data with timestamps)
    return {
      labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
      datasets: [{
        label: 'Average Mood Intensity',
        data: [6.5, 7.2, 6.8, 7.5], // This should come from actual data
        borderColor: 'rgb(236, 72, 153)',
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        tension: 0.3,
        fill: true
      }]
    };
  };

  // Generate Games Performance Chart
  const getGamesChartData = () => {
    const data = filteredData || reportData;
    if (!data?.games?.byGame) return null;

    const gameIds = Object.keys(data.games.byGame);
    const avgScores = gameIds.map(id => parseFloat(data.games.byGame[id].averageScore));
    const gameLabels = gameIds.map(id => getGameName(id));

    return {
      labels: gameLabels,
      datasets: [{
        label: 'Average Score',
        data: avgScores,
        backgroundColor: [
          'rgba(59, 130, 246, 0.7)',
          'rgba(16, 185, 129, 0.7)',
          'rgba(249, 115, 22, 0.7)',
          'rgba(168, 85, 247, 0.7)',
          'rgba(236, 72, 153, 0.7)',
          'rgba(234, 179, 8, 0.7)'
        ],
        borderColor: [
          'rgb(59, 130, 246)',
          'rgb(16, 185, 129)',
          'rgb(249, 115, 22)',
          'rgb(168, 85, 247)',
          'rgb(236, 72, 153)',
          'rgb(234, 179, 8)'
        ],
        borderWidth: 2
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { size: 12, weight: 'bold' },
          padding: 15,
          usePointStyle: true
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          title: (context) => {
            if (context[0].parsed.x) {
              const date = new Date(context[0].parsed.x);
              return format(date, 'MMM dd, yyyy HH:mm');
            }
            return context[0].label;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          displayFormats: { day: 'MMM dd' }
        },
        title: {
          display: true,
          text: 'Date',
          font: { weight: 'bold' }
        },
        grid: { display: false }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Score',
          font: { weight: 'bold' }
        },
        grid: { color: 'rgba(0, 0, 0, 0.05)' }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => `Score: ${context.parsed.y.toFixed(1)}`
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Average Score',
          font: { weight: 'bold' }
        }
      }
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-7xl max-h-[90vh] overflow-y-auto bg-white">
        <DialogHeader>
          <DialogTitle className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <Activity className="w-8 h-8 text-indigo-600" />
            Comprehensive Therapeutic Report
          </DialogTitle>
          <DialogDescription className="text-gray-700 text-lg">
            Complete mental health assessment and analysis for {patientName}
          </DialogDescription>
        </DialogHeader>

        {/* Date Range Selector */}
        {!loading && !error && reportData && (
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
                    setStartDate('');
                    setEndDate('');
                    setFilteredData(null);
                  }}
                  variant="outline"
                  size="sm"
                >
                  Reset
                </Button>
                <div className="ml-auto text-sm text-gray-600">
                  Report Generated: {format(new Date(reportData.patientInfo.reportGeneratedAt), 'MMM dd, yyyy HH:mm')}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-600 mx-auto mb-4"></div>
              <p className="text-gray-600 text-lg">Generating comprehensive report...</p>
              <p className="text-gray-500 text-sm mt-2">Analyzing all patient data sources</p>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
            <p className="text-red-600 font-semibold text-lg">{error}</p>
            <Button onClick={fetchReportData} className="mt-4">
              Retry
            </Button>
          </div>
        ) : !reportData ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📋</div>
            <p className="text-gray-600 text-lg">No report data available</p>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            {/* ALERTS SECTION */}
            {(filteredData || reportData).clinicalInsights?.alerts?.length > 0 && (
              <Card className="border-2 border-red-200 bg-red-50">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-red-900 flex items-center gap-2">
                    <AlertCircle className="w-6 h-6" />
                    Clinical Alerts & Immediate Attention Required
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {(filteredData || reportData).clinicalInsights.alerts.map((alert, idx) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-lg border-2 ${getAlertColor(alert.priority)}`}
                      >
                        <div className="flex items-start gap-3">
                          <AlertCircle className="w-5 h-5 mt-0.5 shrink-0" />
                          <div>
                            <p className="font-semibold">
                              {alert.priority === 'urgent' ? '🚨 URGENT: ' : alert.priority === 'high' ? '⚠️ HIGH: ' : '⚡ '}
                              {alert.message}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* DATA COMPLETENESS SCORE */}
            <Card className="border-2 bg-linear-to-br from-indigo-50 to-purple-50">
              <CardHeader>
                <CardTitle className="text-xl font-bold text-gray-900">Data Completeness</CardTitle>
                <CardDescription>Percentage of available data sources utilized</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="h-8 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full flex items-center justify-center text-white font-bold text-sm transition-all ${
                            reportData.dataCompleteness.completenessScore >= 80 ? 'bg-green-600' :
                            reportData.dataCompleteness.completenessScore >= 50 ? 'bg-yellow-600' :
                            'bg-orange-600'
                          }`}
                          style={{ width: `${reportData.dataCompleteness.completenessScore}%` }}
                        >
                          {reportData.dataCompleteness.completenessScore}%
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <div className={`p-3 rounded-lg border-2 text-center ${reportData.dataCompleteness.hasAssessments ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-300'}`}>
                      <div className="text-2xl mb-1">{reportData.dataCompleteness.hasAssessments ? '✅' : '❌'}</div>
                      <p className="text-xs font-semibold">Assessments</p>
                      <p className="text-xs text-gray-600">{reportData.assessments.totalAssessments} entries</p>
                    </div>
                    
                    <div className={`p-3 rounded-lg border-2 text-center ${reportData.dataCompleteness.hasJournals ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-300'}`}>
                      <div className="text-2xl mb-1">{reportData.dataCompleteness.hasJournals ? '✅' : '❌'}</div>
                      <p className="text-xs font-semibold">Journals</p>
                      <p className="text-xs text-gray-600">{reportData.journals.totalEntries} entries</p>
                    </div>
                    
                    <div className={`p-3 rounded-lg border-2 text-center ${reportData.dataCompleteness.hasMoods ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-300'}`}>
                      <div className="text-2xl mb-1">{reportData.dataCompleteness.hasMoods ? '✅' : '❌'}</div>
                      <p className="text-xs font-semibold">Mood Tracking</p>
                      <p className="text-xs text-gray-600">{reportData.moods.totalEntries} entries</p>
                    </div>
                    
                    <div className={`p-3 rounded-lg border-2 text-center ${reportData.dataCompleteness.hasGames ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-300'}`}>
                      <div className="text-2xl mb-1">{reportData.dataCompleteness.hasGames ? '✅' : '❌'}</div>
                      <p className="text-xs font-semibold">Games</p>
                      <p className="text-xs text-gray-600">{reportData.games.totalPlayed} played</p>
                    </div>
                    
                    <div className={`p-3 rounded-lg border-2 text-center ${reportData.dataCompleteness.hasConversations ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-300'}`}>
                      <div className="text-2xl mb-1">{reportData.dataCompleteness.hasConversations ? '✅' : '❌'}</div>
                      <p className="text-xs font-semibold">Chatbot</p>
                      <p className="text-xs text-gray-600">{reportData.conversations.totalConversations} chats</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* ASSESSMENT SUMMARY */}
            {reportData.dataCompleteness.hasAssessments && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Activity className="w-6 h-6 text-blue-600" />
                    Mental Health Assessments
                  </CardTitle>
                  <CardDescription>Standardized assessment scores and trends</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {Object.entries((filteredData || reportData).assessments.summary).map(([type, data]) => (
                      data.latest && (
                        <Card key={type} className="border-2 hover:shadow-lg transition-shadow">
                          <CardHeader className="pb-3">
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-sm font-bold">{type}</CardTitle>
                              {data.trend && getTrendIcon(data.trend.direction)}
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-2">
                            <div className="text-center">
                              <div className="text-4xl font-bold text-gray-900">{data.latest.score}</div>
                              <Badge className={`mt-2 ${getSeverityColor(data.latest.severity.color)}`}>
                                {data.latest.severity.level}
                              </Badge>
                            </div>
                            {data.trend && (
                              <p className="text-xs text-gray-600 text-center mt-2">
                                {data.trend.direction === 'improving' ? '📉 Improving' : 
                                 data.trend.direction === 'worsening' ? '📈 Worsening' : 
                                 '➡️ Stable'} ({data.trend.change}%)
                              </p>
                            )}
                            <p className="text-xs text-gray-500 text-center">
                              {data.scores.length} assessments completed
                            </p>
                          </CardContent>
                        </Card>
                      )
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ASSESSMENT TIMELINE CHART */}
            {(filteredData || reportData)?.dataCompleteness?.hasAssessments && getAssessmentChartData() && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <BarChart3 className="w-6 h-6 text-indigo-600" />
                    Assessment Score Timeline
                  </CardTitle>
                  <CardDescription>Track mental health assessment scores over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-96">
                    <Line data={getAssessmentChartData()} options={chartOptions} />
                  </div>
                  <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>📊 Chart Interpretation:</strong> Lower scores generally indicate better mental health. 
                      Track trends over time to monitor treatment effectiveness and patient progress.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* GAMES PERFORMANCE CHART */}
            {(filteredData || reportData)?.dataCompleteness?.hasGames && getGamesChartData() && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Gamepad2 className="w-6 h-6 text-green-600" />
                    Cognitive Performance by Game
                  </CardTitle>
                  <CardDescription>Average scores across different cognitive games</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-96">
                    <Bar data={getGamesChartData()} options={barChartOptions} />
                  </div>
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>🎮 Performance Insights:</strong> Higher scores indicate better cognitive performance. 
                      Consistent improvement across games suggests positive cognitive development.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* JOURNAL ANALYSIS */}
            {reportData.dataCompleteness.hasJournals && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <BookOpen className="w-6 h-6 text-purple-600" />
                    Journal & Emotion Analysis
                  </CardTitle>
                  <CardDescription>{(filteredData || reportData).journals.totalEntries} journal entries analyzed</CardDescription>
                </CardHeader>
                <CardContent>
<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Mood Distribution */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Mood Distribution</h4>
                      <div className="space-y-2">
                        {Object.entries((filteredData || reportData).journals.recentMoods).map(([mood, count]) => (
                          <div key={mood} className="flex items-center gap-2">
                            <div className="w-24 text-sm capitalize">{mood}</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                              <div
                                className="bg-linear-to-r from-purple-500 to-pink-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{
                                  width: `${(count / (filteredData || reportData).journals.totalEntries) * 100}%`,
                                }}
                              >
                                {count}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Cognitive Distortions */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Top Cognitive Distortions</h4>
                      <div className="space-y-2">
                        {(filteredData || reportData).journals.cognitiveDistortions.slice(0, 5).map((distortion, idx) => (
                          <div key={idx} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-900">{distortion.type}</span>
                              <Badge className="bg-yellow-600 text-white">{distortion.count}x</Badge>
                            </div>
                          </div>
                        ))}
                        {(filteredData || reportData).journals.cognitiveDistortions.length === 0 && (
                          <p className="text-sm text-gray-500 italic">No cognitive distortions detected</p>
                        )}
                      </div>
                    </div>

                    {/* Risk Indicators */}
                    {(filteredData || reportData).journals.riskIndicators.length > 0 && (
                      <div className="md:col-span-2">
                        <h4 className="font-semibold text-red-900 mb-3 flex items-center gap-2">
                          <AlertCircle className="w-5 h-5" />
                          Risk Indicators Flagged
                        </h4>
                        <div className="space-y-2">
                          {(filteredData || reportData).journals.riskIndicators.slice(0, 3).map((risk, idx) => (
                            <div key={idx} className="p-3 bg-red-50 border-2 border-red-200 rounded-lg">
                              <div className="flex items-start gap-2">
                                <Badge className={`${risk.level === 'high' ? 'bg-red-600' : 'bg-orange-600'} text-white`}>
                                  {risk.level.toUpperCase()}
                                </Badge>
                                <div className="flex-1">
                                  <p className="text-sm text-gray-700">{risk.excerpt}</p>
                                  <p className="text-xs text-gray-500 mt-1">
                                    {risk.date 
                                      ? new Date(risk.date).toLocaleDateString('en-US', { 
                                          year: 'numeric', 
                                          month: 'short', 
                                          day: 'numeric' 
                                        })
                                      : 'Date not available'}
                                  </p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* MOOD TRACKING */}
            {reportData.dataCompleteness.hasMoods && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Heart className="w-6 h-6 text-pink-600" />
                    Daily Mood Tracking
                  </CardTitle>
                  <CardDescription>{reportData.moods.totalEntries} mood entries logged</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-linear-to-br from-pink-50 to-rose-50">
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <p className="text-sm text-gray-600 mb-1">Average Intensity</p>
                          <p className="text-4xl font-bold text-pink-700">{reportData.moods.averageIntensity}<span className="text-lg">/10</span></p>
                        </div>
                      </CardContent>
                    </Card>

                    {reportData.moods.recentTrend && (
                      <Card className="bg-linear-to-br from-blue-50 to-indigo-50">
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <p className="text-sm text-gray-600 mb-1">Recent Trend</p>
                            <div className="flex items-center justify-center gap-2">
                              {getTrendIcon(reportData.moods.recentTrend.direction)}
                              <p className="text-2xl font-bold text-indigo-700 capitalize">
                                {reportData.moods.recentTrend.direction}
                              </p>
                            </div>
                            <p className="text-xs text-gray-500 mt-1">{reportData.moods.recentTrend.change}% change</p>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    <Card className="bg-linear-to-br from-green-50 to-emerald-50">
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <p className="text-sm text-gray-600 mb-1">Most Common Mood</p>
                          <p className="text-3xl font-bold text-green-700 capitalize">
                            {Object.entries(reportData.moods.distribution).sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A'}
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* GAMES & COGNITIVE METRICS */}
            {reportData.dataCompleteness.hasGames && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Gamepad2 className="w-6 h-6 text-green-600" />
                    Cognitive Games & Performance
                  </CardTitle>
                  <CardDescription>{reportData.games.totalPlayed} games completed</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <Card className="bg-green-50">
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Average Accuracy</p>
                        <p className="text-3xl font-bold text-green-700">
                          {reportData.games.cognitiveMetrics.averageAccuracy}%
                        </p>
                      </CardContent>
                    </Card>

                    <Card className="bg-blue-50">
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Avg Reaction Time</p>
                        <p className="text-3xl font-bold text-blue-700">
                          {reportData.games.cognitiveMetrics.averageReactionTime}ms
                        </p>
                      </CardContent>
                    </Card>

                    <Card className="bg-purple-50">
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Games Played</p>
                        <p className="text-3xl font-bold text-purple-700">
                          {Object.keys(reportData.games.byGame).length}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Performance by Game</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {Object.entries(reportData.games.byGame).map(([gameId, data]) => {
                        const game = getGameById(gameId);
                        const domainColors = getDomainColors(getCognitiveDomain(gameId));
                        return (
                          <div key={gameId} className={`p-4 ${domainColors.bg} border-2 ${domainColors.border} rounded-lg hover:shadow-md transition-shadow`}>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <span className="text-xl">{game?.emoji || '🎮'}</span>
                                <span className="text-sm font-semibold text-gray-900">{getGameName(gameId)}</span>
                              </div>
                              <Badge className="bg-gray-700 text-white">{data.count}x</Badge>
                            </div>
                            <p className="text-xs text-gray-600 mb-1">Avg Score: <span className="font-bold text-lg">{data.averageScore}</span></p>
                            <p className="text-xs text-gray-500">{game?.cognitiveDomain || 'Cognitive'}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Cognitive Domain Analysis */}
                  <div className="mt-6">
                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Brain className="w-5 h-5 text-indigo-600" />
                      Performance by Cognitive Domain
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(
                        Object.entries(reportData.games.byGame).reduce((acc, [gameId, data]) => {
                          const domain = getCognitiveDomain(gameId);
                          if (!acc[domain]) {
                            acc[domain] = { totalScore: 0, count: 0, games: [] };
                          }
                          acc[domain].totalScore += parseFloat(data.averageScore);
                          acc[domain].count += 1;
                          acc[domain].games.push({ id: gameId, ...data });
                          return acc;
                        }, {})
                      ).map(([domain, stats]) => {
                        const avgScore = (stats.totalScore / stats.count).toFixed(1);
                        const domainColors = getDomainColors(domain);
                        return (
                          <div key={domain} className={`p-4 ${domainColors.bg} border-2 ${domainColors.border} rounded-lg`}>
                            <div className="flex items-center justify-between mb-3">
                              <h5 className={`font-bold ${domainColors.text}`}>{domain}</h5>
                              <Badge className="bg-white text-gray-900 border border-gray-300">{stats.count} games</Badge>
                            </div>
                            <div className="mb-3">
                              <p className="text-xs text-gray-600 mb-1">Average Performance</p>
                              <div className="flex items-center gap-2">
                                <div className="flex-1 bg-white rounded-full h-3 overflow-hidden border border-gray-300">
                                  <div
                                    className="h-full bg-gradient-to-r from-green-400 to-green-600"
                                    style={{ width: `${Math.min(avgScore, 100)}%` }}
                                  />
                                </div>
                                <span className="text-lg font-bold text-gray-900">{avgScore}</span>
                              </div>
                            </div>
                            <div className="space-y-1">
                              {stats.games.map(game => {
                                const gameInfo = getGameById(game.id);
                                return (
                                  <div key={game.id} className="flex items-center justify-between text-xs">
                                    <span className="text-gray-700">{gameInfo?.emoji} {getGameName(game.id)}</span>
                                    <span className="font-semibold text-gray-900">{game.averageScore}</span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* CHATBOT CONVERSATIONS */}
            {reportData.dataCompleteness.hasConversations && (
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <MessageSquare className="w-6 h-6 text-indigo-600" />
                    Chatbot Engagement & Topics
                  </CardTitle>
                  <CardDescription>{reportData.conversations.totalConversations} conversations with AI therapist</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <Card className="bg-indigo-50">
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Total Messages</p>
                        <p className="text-3xl font-bold text-indigo-700">{reportData.conversations.totalMessages}</p>
                      </CardContent>
                    </Card>

                    <Card className="bg-purple-50">
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Avg Messages/Chat</p>
                        <p className="text-3xl font-bold text-purple-700">{reportData.conversations.averageMessagesPerConversation}</p>
                      </CardContent>
                    </Card>

                    <Card className={`${reportData.conversations.engagementLevel === 'high' ? 'bg-green-50' : reportData.conversations.engagementLevel === 'medium' ? 'bg-yellow-50' : 'bg-gray-50'}`}>
                      <CardContent className="pt-6 text-center">
                        <p className="text-sm text-gray-600 mb-1">Engagement Level</p>
                        <p className={`text-2xl font-bold capitalize ${reportData.conversations.engagementLevel === 'high' ? 'text-green-700' : reportData.conversations.engagementLevel === 'medium' ? 'text-yellow-700' : 'text-gray-700'}`}>
                          {reportData.conversations.engagementLevel}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Recent Conversation Topics</h4>
                    <div className="space-y-2">
                      {reportData.conversations.recentTopics.slice(0, 5).map((topic, idx) => (
                        <div key={idx} className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                          <div className="flex items-start justify-between gap-2">
                            <p className="text-sm text-gray-900 flex-1">{topic.topic}</p>
                            <div className="text-right">
                              <Badge className="bg-indigo-600 text-white text-xs">{topic.messageCount} msgs</Badge>
                              <p className="text-xs text-gray-500 mt-1">
                                {topic.date 
                                  ? new Date(topic.date).toLocaleDateString('en-US', { 
                                      year: 'numeric', 
                                      month: 'short', 
                                      day: 'numeric' 
                                    })
                                  : 'Date not available'}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ANALYTICS SECTION - Mood/Sentiment/Risk Trends */}
            {(filteredData || reportData).journals && (
              <>
                {/* Mood Over Time */}
                {(filteredData || reportData).journals.emotionTrends && (filteredData || reportData).journals.emotionTrends.length > 0 && (
                  <Card className="border-2">
                    <CardHeader>
                      <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                        <Activity className="w-6 h-6 text-blue-600" />
                        Emotional Pattern Timeline
                      </CardTitle>
                      <CardDescription>Emotional intensity and patterns tracked over the selected period</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={(filteredData || reportData).journals.emotionTrends}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <RechartsLine
                            type="monotone"
                            dataKey="intensity"
                            stroke="#3b82f6"
                            name="Emotional Intensity"
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                )}

                {/* Risk Indicators Timeline */}
                {(filteredData || reportData).journals.riskIndicators && (filteredData || reportData).journals.riskIndicators.length > 0 && (
                  <Card className="border-2">
                    <CardHeader>
                      <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                        <AlertCircle className="w-6 h-6 text-red-600" />
                        Risk Level Timeline
                      </CardTitle>
                      <CardDescription>Risk assessment trends based on journal entries and indicators</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <Card className="bg-red-50 border border-red-200">
                            <CardContent className="pt-6">
                              <div className="text-center">
                                <p className="text-sm text-red-700 font-semibold mb-2">High Risk Indicators</p>
                                <p className="text-3xl font-bold text-red-600">
                                  {((filteredData || reportData).journals.riskIndicators || []).filter(r => r.level === 'high').length}
                                </p>
                              </div>
                            </CardContent>
                          </Card>
                          <Card className="bg-orange-50 border border-orange-200">
                            <CardContent className="pt-6">
                              <div className="text-center">
                                <p className="text-sm text-orange-700 font-semibold mb-2">Medium Risk Indicators</p>
                                <p className="text-3xl font-bold text-orange-600">
                                  {((filteredData || reportData).journals.riskIndicators || []).filter(r => r.level === 'medium').length}
                                </p>
                              </div>
                            </CardContent>
                          </Card>
                          <Card className="bg-yellow-50 border border-yellow-200">
                            <CardContent className="pt-6">
                              <div className="text-center">
                                <p className="text-sm text-yellow-700 font-semibold mb-2">Low Risk Indicators</p>
                                <p className="text-3xl font-bold text-yellow-600">
                                  {((filteredData || reportData).journals.riskIndicators || []).filter(r => r.level === 'low').length}
                                </p>
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                        
                        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                          <p className="text-sm font-semibold text-red-900 mb-2">Recent Risk Incidents:</p>
                          <div className="space-y-2">
                            {((filteredData || reportData).journals.riskIndicators || []).slice(0, 3).map((risk, idx) => (
                              <div key={idx} className="text-sm text-red-800">
                                <span className={`inline-block px-2 py-1 rounded text-xs font-semibold mr-2 ${
                                  risk.level === 'high' ? 'bg-red-600 text-white' : 
                                  risk.level === 'medium' ? 'bg-orange-600 text-white' :
                                  'bg-yellow-600 text-white'
                                }`}>
                                  {risk.level.toUpperCase()}
                                </span>
                                {risk.excerpt || risk.type}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {/* CLINICAL INSIGHTS */}
            {reportData.clinicalInsights && (
              <Card className="border-2 border-indigo-200 bg-linear-to-br from-indigo-50 to-purple-50">
                <CardHeader>
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Brain className="w-6 h-6 text-indigo-600" />
                    Clinical Insights & Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Insights */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        💡 Key Insights
                      </h4>
                      <div className="space-y-2">
                        {reportData.clinicalInsights.insights.length > 0 ? (
                          reportData.clinicalInsights.insights.map((insight, idx) => (
                            <div key={idx} className="p-3 bg-white border border-indigo-200 rounded-lg">
                              <p className="text-sm text-gray-800">{insight}</p>
                            </div>
                          ))
                        ) : (
                          <p className="text-sm text-gray-500 italic">No insights generated yet</p>
                        )}
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        📋 Treatment Recommendations
                      </h4>
                      <div className="space-y-2">
                        {reportData.clinicalInsights.recommendations.length > 0 ? (
                          reportData.clinicalInsights.recommendations.map((rec, idx) => (
                            <div key={idx} className="p-3 bg-white border border-purple-200 rounded-lg">
                              <p className="text-sm text-gray-800">• {rec}</p>
                            </div>
                          ))
                        ) : (
                          <p className="text-sm text-gray-500 italic">No specific recommendations at this time</p>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Action Buttons */}
            <div className="flex justify-end gap-3 pt-6 border-t">
              <Button variant="outline" onClick={onClose} className="font-semibold">
                Close
              </Button>
              <Button 
                onClick={() => window.print()} 
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold"
              >
                🖨️ Print Report
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
