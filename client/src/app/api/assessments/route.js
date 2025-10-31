import { NextResponse } from 'next/server';
import { getAuth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import Assessment from '@/models/Assessment';

export async function POST(req) {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    const assessmentData = await req.json();
    
    try {
      await connectDB();
    } catch (dbError) {
      console.error('Database connection error:', dbError);
      return new NextResponse(
        JSON.stringify({ error: 'Database connection failed', details: dbError.message }), 
        { status: 500 }
      );
    }

    // Handle both old format (gad7, phq9) and new format (multiple assessments)
    let assessment;
    if (assessmentData.gad7 || assessmentData.phq9) {
      // Old format - backward compatibility
      assessment = new Assessment({
        userId,
        gad7: assessmentData.gad7,
        phq9: assessmentData.phq9,
      });
    } else {
      // New format - multiple assessment types
      assessment = new Assessment({
        userId,
        ...assessmentData
      });
    }

    try {
      await assessment.save();
      return NextResponse.json(assessment);
    } catch (saveError) {
      console.error('Error saving assessment:', saveError);
      return new NextResponse(
        JSON.stringify({ error: 'Failed to save assessment', details: saveError.message }), 
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Error in assessment route:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal server error', details: error.message }), 
      { status: 500 }
    );
  }
}

export async function GET(req) {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    await connectDB();
    const assessments = await Assessment.find({ userId }).sort({ date: -1 });

    return NextResponse.json(assessments);
  } catch (error) {
    console.error('Error fetching assessments:', error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}