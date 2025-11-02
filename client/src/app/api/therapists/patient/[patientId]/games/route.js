import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import GameResult from '@/models/GameResult';
import { computeGameAnalytics } from '@/lib/analysis';

// GET /api/therapists/patient/[patientId]/games
export async function GET(request, { params }) {
  try {
    const { userId, isAuthenticated } = await auth(request);
    if (!isAuthenticated || !userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { patientId } = await params;

    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    const patient = await Patient.findById(patientId).select('clerkId medicalInfo.assignedTherapist');
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const assigned = patient.medicalInfo?.assignedTherapist?.toString();
    if (!assigned || assigned !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    // Optional filters
    const { searchParams } = new URL(request.url);
    const since = searchParams.get('since');
    const until = searchParams.get('until');
    const gameId = searchParams.get('gameId');

    const query = { userId: patient.clerkId };
    if (gameId) query.gameId = gameId;
    if (since || until) {
      query.createdAt = {};
      if (since) query.createdAt.$gte = new Date(since);
      if (until) query.createdAt.$lte = new Date(until);
    }

    const results = await GameResult.find(query).sort({ createdAt: 1 }).lean();
    const analytics = computeGameAnalytics(results);

    return NextResponse.json({ patientId, ...analytics });
  } catch (error) {
    console.error('Therapist games analytics error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
