import { getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '../../../../../lib/mongodb';
import Assessment from '../../../../../models/Assessment';
import Patient from '../../../../../models/Patient';

export async function GET(request, { params }) {
  try {
    const auth = getAuth(request);
    const { userId } = auth;

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    // Await params in Next.js 15+
    const { patientId } = await params;
    console.log('🔍 Assessment Report - Received patientId:', patientId);

    // Get the patient to find their userId (Clerk ID)
    const patient = await Patient.findById(patientId).populate('userId');
    
    if (!patient) {
      console.log('❌ Patient not found for patientId:', patientId);
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const patientClerkId = patient.userId?.clerkId;
    console.log('✅ Patient found:', patient.userId?.firstName, patient.userId?.lastName);
    console.log('   Clerk ID:', patientClerkId);
    
    if (!patientClerkId) {
      console.log('⚠️ No Clerk ID found for patient');
      return NextResponse.json({ 
        assessments: [],
        summary: {
          totalAssessments: 0,
          latestScores: {},
          hasData: false
        }
      });
    }

    // Get all assessments for this patient's Clerk ID
    const assessments = await Assessment.find({ userId: patientClerkId })
      .sort({ date: -1 })
      .lean();
    console.log('📊 Found', assessments.length, 'assessments for this patient');

    if (!assessments || assessments.length === 0) {
      return NextResponse.json({ 
        assessments: [],
        summary: {
          totalAssessments: 0,
          latestScores: {},
          hasData: false
        }
      });
    }

    // Transform assessments to match expected format
    const assessmentsByType = {
      'GAD-7': [],
      'PHQ-9': [],
      'PSS-10': [],
      'ISI': []
    };

    const latestScores = {};

    assessments.forEach(assessment => {
      // GAD-7 (Anxiety)
      if (assessment.gad7 && assessment.gad7.score !== undefined) {
        assessmentsByType['GAD-7'].push({
          score: assessment.gad7.score,
          date: assessment.date,
          responses: assessment.gad7.answers
        });
        if (!latestScores['GAD-7']) {
          latestScores['GAD-7'] = {
            score: assessment.gad7.score,
            date: assessment.date,
            severity: getSeverityLevel('GAD-7', assessment.gad7.score)
          };
        }
      }

      // PHQ-9 (Depression)
      if (assessment.phq9 && assessment.phq9.score !== undefined) {
        assessmentsByType['PHQ-9'].push({
          score: assessment.phq9.score,
          date: assessment.date,
          responses: assessment.phq9.answers
        });
        if (!latestScores['PHQ-9']) {
          latestScores['PHQ-9'] = {
            score: assessment.phq9.score,
            date: assessment.date,
            severity: getSeverityLevel('PHQ-9', assessment.phq9.score)
          };
        }
      }

      // PSS-10 (Stress)
      if (assessment.stress && assessment.stress.score !== undefined) {
        assessmentsByType['PSS-10'].push({
          score: assessment.stress.score,
          date: assessment.date,
          responses: assessment.stress.answers
        });
        if (!latestScores['PSS-10']) {
          latestScores['PSS-10'] = {
            score: assessment.stress.score,
            date: assessment.date,
            severity: getSeverityLevel('PSS-10', assessment.stress.score)
          };
        }
      }

      // ISI (Insomnia/Sleep)
      if (assessment.sleep && assessment.sleep.score !== undefined) {
        assessmentsByType['ISI'].push({
          score: assessment.sleep.score,
          date: assessment.date,
          responses: assessment.sleep.answers
        });
        if (!latestScores['ISI']) {
          latestScores['ISI'] = {
            score: assessment.sleep.score,
            date: assessment.date,
            severity: getSeverityLevel('ISI', assessment.sleep.score)
          };
        }
      }
    });

    const totalAssessments = Object.values(assessmentsByType).reduce((sum, arr) => sum + arr.length, 0);

    return NextResponse.json({
      assessments: assessmentsByType,
      summary: {
        totalAssessments,
        latestScores,
        hasData: totalAssessments > 0
      }
    });
  } catch (error) {
    console.error('Error fetching patient assessments:', error);
    return NextResponse.json({ 
      error: 'Internal Server Error',
      message: error.message 
    }, { status: 500 });
  }
}

// Helper function to determine severity level
function getSeverityLevel(type, score) {
  switch(type) {
    case 'GAD-7':
      if (score <= 4) return { level: 'Minimal', color: 'green' };
      if (score <= 9) return { level: 'Mild', color: 'yellow' };
      if (score <= 14) return { level: 'Moderate', color: 'orange' };
      return { level: 'Severe', color: 'red' };
    
    case 'PHQ-9':
      if (score <= 4) return { level: 'Minimal', color: 'green' };
      if (score <= 9) return { level: 'Mild', color: 'yellow' };
      if (score <= 14) return { level: 'Moderate', color: 'orange' };
      if (score <= 19) return { level: 'Moderately Severe', color: 'red' };
      return { level: 'Severe', color: 'red' };
    
    case 'PSS-10':
      if (score <= 13) return { level: 'Low Stress', color: 'green' };
      if (score <= 26) return { level: 'Moderate Stress', color: 'orange' };
      return { level: 'High Stress', color: 'red' };
    
    case 'ISI':
      if (score <= 7) return { level: 'No Insomnia', color: 'green' };
      if (score <= 14) return { level: 'Subthreshold', color: 'yellow' };
      if (score <= 21) return { level: 'Moderate', color: 'orange' };
      return { level: 'Severe', color: 'red' };
    
    default:
      return { level: 'Unknown', color: 'gray' };
  }
}
