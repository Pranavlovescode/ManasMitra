import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import Assessment from '@/models/Assessment';
import Journal from '@/models/Journal';
import GameResult from '@/models/GameResult';
import Conversation from '@/models/Conversation';
import mongoose from 'mongoose';

// Mood schema (inline as it's not a separate model file)
const MoodSchema = new mongoose.Schema({
  userId: { type: String, required: true, index: true },
  mood: { type: String, required: true, enum: ['sad', 'neutral', 'happy', 'excited', 'loved'] },
  intensity: { type: Number, required: true, min: 1, max: 10 },
  notes: { type: String, default: '' },
  date: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
}, { timestamps: true });

const Mood = mongoose.models.Mood || mongoose.model('Mood', MoodSchema);

// Helper function to get severity level
function getSeverityLevel(type, score) {
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
}

// Helper function to analyze trends
function analyzeTrends(dataPoints) {
  if (!dataPoints || dataPoints.length < 2) {
    return { direction: 'insufficient_data', change: 0 };
  }

  const recent = dataPoints.slice(0, Math.min(3, dataPoints.length));
  const older = dataPoints.slice(-Math.min(3, dataPoints.length));

  const recentAvg = recent.reduce((sum, p) => sum + p.score, 0) / recent.length;
  const olderAvg = older.reduce((sum, p) => sum + p.score, 0) / older.length;

  const change = ((recentAvg - olderAvg) / olderAvg) * 100;

  let direction = 'stable';
  if (change > 10) direction = 'worsening';
  else if (change < -10) direction = 'improving';

  return { direction, change: Math.abs(change).toFixed(1) };
}

// GET /api/therapists/patient/[patientId]/comprehensive-report
export async function GET(request, context) {
  try {
    const { userId } = await auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    // In Next.js 15+, params might need to be awaited
    const params = await context.params;
    const { patientId } = params;
    
    console.log('Comprehensive Report - Patient ID:', patientId);
    console.log('Comprehensive Report - User ID:', userId);

    // Get therapist user
    const therapistUser = await User.findOne({ clerkId: userId });
    if (!therapistUser || therapistUser.role !== 'therapist') {
      return NextResponse.json({ error: 'Forbidden: only therapists can access this endpoint' }, { status: 403 });
    }

    // Get patient and verify assignment
    const patient = await Patient.findById(patientId).populate('userId').lean();
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    // Verify therapist is assigned to this patient
    const assignedTherapistId = patient.medicalInfo?.assignedTherapist || patient.status?.assignedTherapist;
    if (assignedTherapistId && assignedTherapistId.toString() !== therapistUser._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    const patientClerkId = patient.clerkId;

    // Fetch all data in parallel
    const [assessments, journals, moods, games, conversations] = await Promise.all([
      Assessment.find({ userId: patientClerkId }).sort({ date: -1 }).lean(),
      Journal.find({ userId: patientClerkId }).sort({ createdAt: -1 }).lean(),
      Mood.find({ userId: patientClerkId }).sort({ timestamp: -1 }).lean(),
      GameResult.find({ userId: patientClerkId }).sort({ createdAt: -1 }).lean(),
      Conversation.find({ userId: patientClerkId }).sort({ createdAt: -1 }).lean()
    ]);

    // ============ ASSESSMENT ANALYSIS ============
    const assessmentSummary = {
      'GAD-7': { scores: [], latest: null, trend: null },
      'PHQ-9': { scores: [], latest: null, trend: null },
      'PSS-10': { scores: [], latest: null, trend: null },
      'ISI': { scores: [], latest: null, trend: null }
    };

    assessments.forEach(assessment => {
      if (assessment.gad7 && assessment.gad7.score !== undefined) {
        assessmentSummary['GAD-7'].scores.push({
          score: assessment.gad7.score,
          date: assessment.date
        });
      }
      if (assessment.phq9 && assessment.phq9.score !== undefined) {
        assessmentSummary['PHQ-9'].scores.push({
          score: assessment.phq9.score,
          date: assessment.date
        });
      }
      if (assessment.stress && assessment.stress.score !== undefined) {
        assessmentSummary['PSS-10'].scores.push({
          score: assessment.stress.score,
          date: assessment.date
        });
      }
      if (assessment.sleep && assessment.sleep.score !== undefined) {
        assessmentSummary['ISI'].scores.push({
          score: assessment.sleep.score,
          date: assessment.date
        });
      }
    });

    // Calculate latest and trends
    Object.keys(assessmentSummary).forEach(type => {
      const data = assessmentSummary[type];
      if (data.scores.length > 0) {
        data.latest = {
          ...data.scores[0],
          severity: getSeverityLevel(type, data.scores[0].score)
        };
        data.trend = analyzeTrends(data.scores);
      }
    });

    // ============ JOURNAL ANALYSIS ============
    const journalAnalysis = {
      totalEntries: journals.length,
      recentMoods: {},
      emotionTrends: [],
      cognitiveDistortions: [],
      riskIndicators: [],
      keyThemes: []
    };

    journals.forEach(journal => {
      // Mood distribution
      if (journal.mood) {
        journalAnalysis.recentMoods[journal.mood] = (journalAnalysis.recentMoods[journal.mood] || 0) + 1;
      }

      // Emotion from analysis
      if (journal.analysis?.contentAnalysis?.emotion) {
        journalAnalysis.emotionTrends.push({
          emotion: journal.analysis.contentAnalysis.emotion,
          score: journal.analysis.contentAnalysis.emotionScore || 0,
          date: journal.createdAt
        });
      }

      // Cognitive distortions
      if (journal.analysis?.contentAnalysis?.distortions) {
        journal.analysis.contentAnalysis.distortions.forEach(distortion => {
          const existing = journalAnalysis.cognitiveDistortions.find(d => d.type === distortion);
          if (existing) {
            existing.count++;
          } else {
            journalAnalysis.cognitiveDistortions.push({ type: distortion, count: 1 });
          }
        });
      }

      // Risk indicators
      if (journal.analysis?.contentAnalysis?.risk && journal.analysis.contentAnalysis.risk !== 'low') {
        journalAnalysis.riskIndicators.push({
          level: journal.analysis.contentAnalysis.risk,
          score: journal.analysis.contentAnalysis.riskScore || 0,
          date: journal.createdAt,
          excerpt: journal.content.substring(0, 100) + '...'
        });
      }
    });

    // Sort cognitive distortions by frequency
    journalAnalysis.cognitiveDistortions.sort((a, b) => b.count - a.count);

    // ============ MOOD TRACKING ANALYSIS ============
    const moodAnalysis = {
      totalEntries: moods.length,
      distribution: {},
      averageIntensity: 0,
      recentTrend: null
    };

    moods.forEach(mood => {
      moodAnalysis.distribution[mood.mood] = (moodAnalysis.distribution[mood.mood] || 0) + 1;
    });

    if (moods.length > 0) {
      const totalIntensity = moods.reduce((sum, m) => sum + (m.intensity || 0), 0);
      moodAnalysis.averageIntensity = (totalIntensity / moods.length).toFixed(1);

      // Recent trend (last 7 vs previous 7)
      if (moods.length >= 7) {
        const recent7 = moods.slice(0, 7);
        const previous7 = moods.slice(7, 14);
        
        const recentAvg = recent7.reduce((sum, m) => sum + (m.intensity || 0), 0) / 7;
        const previousAvg = previous7.length > 0 
          ? previous7.reduce((sum, m) => sum + (m.intensity || 0), 0) / previous7.length
          : recentAvg;

        const change = ((recentAvg - previousAvg) / previousAvg) * 100;
        moodAnalysis.recentTrend = {
          direction: change > 5 ? 'improving' : change < -5 ? 'declining' : 'stable',
          change: Math.abs(change).toFixed(1)
        };
      }
    }

    // ============ GAMES ANALYSIS ============
    const gamesAnalysis = {
      totalPlayed: games.length,
      byGame: {},
      cognitiveMetrics: {
        averageAccuracy: 0,
        averageReactionTime: 0,
        improvementTrend: null
      }
    };

    games.forEach(game => {
      if (!gamesAnalysis.byGame[game.gameId]) {
        gamesAnalysis.byGame[game.gameId] = {
          count: 0,
          averageScore: 0,
          scores: []
        };
      }
      gamesAnalysis.byGame[game.gameId].count++;
      gamesAnalysis.byGame[game.gameId].scores.push(game.score || 0);
    });

    // Calculate averages
    Object.keys(gamesAnalysis.byGame).forEach(gameId => {
      const gameData = gamesAnalysis.byGame[gameId];
      gameData.averageScore = (gameData.scores.reduce((sum, s) => sum + s, 0) / gameData.count).toFixed(1);
    });

    // Overall cognitive metrics
    let totalAccuracy = 0;
    let totalReactionTime = 0;
    let metricsCount = 0;

    games.forEach(game => {
      if (game.metrics?.accuracy) {
        totalAccuracy += game.metrics.accuracy;
        metricsCount++;
      }
      if (game.metrics?.avgReactionMs) {
        totalReactionTime += game.metrics.avgReactionMs;
      }
    });

    if (metricsCount > 0) {
      gamesAnalysis.cognitiveMetrics.averageAccuracy = ((totalAccuracy / metricsCount) * 100).toFixed(1);
      gamesAnalysis.cognitiveMetrics.averageReactionTime = (totalReactionTime / metricsCount).toFixed(0);
    }

    // ============ CHATBOT CONVERSATION ANALYSIS ============
    const conversationAnalysis = {
      totalConversations: conversations.length,
      totalMessages: 0,
      averageMessagesPerConversation: 0,
      recentTopics: [],
      engagementLevel: 'low'
    };

    conversations.forEach(conv => {
      const messageCount = conv.messages?.length || 0;
      conversationAnalysis.totalMessages += messageCount;
      
      // Extract topics from first user message
      const firstUserMessage = conv.messages?.find(m => m.role === 'user');
      if (firstUserMessage) {
        conversationAnalysis.recentTopics.push({
          topic: conv.title || firstUserMessage.content.substring(0, 50) + '...',
          date: conv.createdAt,
          messageCount
        });
      }
    });

    if (conversations.length > 0) {
      conversationAnalysis.averageMessagesPerConversation = 
        (conversationAnalysis.totalMessages / conversations.length).toFixed(1);
      
      // Engagement level based on activity
      if (conversationAnalysis.totalMessages > 50) {
        conversationAnalysis.engagementLevel = 'high';
      } else if (conversationAnalysis.totalMessages > 20) {
        conversationAnalysis.engagementLevel = 'medium';
      }
    }

    // ============ THERAPEUTIC INSIGHTS & RECOMMENDATIONS ============
    const insights = [];
    const recommendations = [];
    const alerts = [];

    // Risk assessment
    if (journalAnalysis.riskIndicators.length > 0) {
      const highRisk = journalAnalysis.riskIndicators.filter(r => r.level === 'high').length;
      if (highRisk > 0) {
        alerts.push({
          type: 'high',
          message: `${highRisk} journal entries flagged with high risk indicators`,
          priority: 'urgent'
        });
        recommendations.push('Immediate clinical assessment recommended for risk evaluation');
      }
    }

    // Assessment trends
    Object.keys(assessmentSummary).forEach(type => {
      const data = assessmentSummary[type];
      if (data.latest && data.trend) {
        if (data.latest.severity.color === 'red') {
          alerts.push({
            type: 'high',
            message: `${type} shows ${data.latest.severity.level} level (Score: ${data.latest.score})`,
            priority: 'high'
          });
        }
        
        if (data.trend.direction === 'worsening') {
          insights.push(`${type} scores have worsened by ${data.trend.change}% recently`);
          recommendations.push(`Consider adjusting treatment approach for ${type.includes('GAD') ? 'anxiety' : type.includes('PHQ') ? 'depression' : 'stress/sleep'}`);
        } else if (data.trend.direction === 'improving') {
          insights.push(`${type} scores have improved by ${data.trend.change}% - current treatment appears effective`);
        }
      }
    });

    // Cognitive distortions
    if (journalAnalysis.cognitiveDistortions.length > 0) {
      const topDistortion = journalAnalysis.cognitiveDistortions[0];
      insights.push(`Most common cognitive distortion: ${topDistortion.type} (${topDistortion.count} occurrences)`);
      recommendations.push(`Focus CBT interventions on addressing ${topDistortion.type} patterns`);
    }

    // Mood patterns
    if (moodAnalysis.recentTrend) {
      if (moodAnalysis.recentTrend.direction === 'declining') {
        alerts.push({
          type: 'medium',
          message: `Mood intensity declining by ${moodAnalysis.recentTrend.change}%`,
          priority: 'medium'
        });
      }
    }

    // Engagement
    if (conversationAnalysis.engagementLevel === 'high') {
      insights.push('High engagement with chatbot therapy - patient is actively seeking support');
    } else if (conversationAnalysis.engagementLevel === 'low' && journals.length < 5) {
      insights.push('Low engagement across digital tools - consider strategies to increase therapeutic engagement');
      recommendations.push('Explore barriers to engagement and provide additional support/encouragement');
    }

    // ============ COMPILE COMPREHENSIVE REPORT ============
    const report = {
      patientInfo: {
        id: patient._id,
        name: `${patient.userId?.firstName} ${patient.userId?.lastName}`,
        age: patient.age,
        gender: patient.gender,
        reportGeneratedAt: new Date()
      },
      assessments: {
        summary: assessmentSummary,
        totalAssessments: assessments.length
      },
      journals: journalAnalysis,
      moods: moodAnalysis,
      games: gamesAnalysis,
      conversations: conversationAnalysis,
      clinicalInsights: {
        insights,
        recommendations,
        alerts
      },
      dataCompleteness: {
        hasAssessments: assessments.length > 0,
        hasJournals: journals.length > 0,
        hasMoods: moods.length > 0,
        hasGames: games.length > 0,
        hasConversations: conversations.length > 0,
        completenessScore: [
          assessments.length > 0,
          journals.length > 0,
          moods.length > 0,
          games.length > 0,
          conversations.length > 0
        ].filter(Boolean).length * 20 // 0-100%
      }
    };

    return NextResponse.json(report);

  } catch (error) {
    console.error('Comprehensive report error:', error);
    return NextResponse.json({ 
      error: 'Internal Server Error',
      message: error.message 
    }, { status: 500 });
  }
}
